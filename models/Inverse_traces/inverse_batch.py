"""File consists InverseBatch class."""
import sys
import numpy as np
import torch

from sklearn.linear_model import LinearRegression

sys.path.append('../..')
from seismicpro.batchflow import action, inbatch_parallel # pylint: disable=wrong-import-position
from seismicpro.src import SeismicBatch # pylint: disable=wrong-import-position


def predict(data, model):
    """Predict first break piking on given data."""
    model.eval()
    with torch.no_grad():
        pred = model(data).cpu().detach().numpy().reshape(2, -1)
    arr = np.argmax(pred, axis=0)
    arr = np.insert(arr, 0, 0)
    arr = np.append(arr, 0)
    plus_one = np.argwhere((np.diff(arr)) == 1).flatten()
    minus_one = np.argwhere((np.diff(arr)) == -1).flatten()
    blocks = minus_one - plus_one
    return plus_one[np.argmax(blocks)]

def crop_trace(trace, pred, size):
    """"Crop traces near ```pred``` value with shape equal to ```size```."""
    lenght = len(trace)
    if lenght < pred + int(size/2):
        left = lenght - size
        right = lenght
    elif pred - int(size/2) < 0:
        left = 0
        right = size
    else:
        left = pred - int(size/2)
        right = pred + int(size/2)
    return np.array(trace[left: right])

def calculate_corr(corr_vec, field, i, neig, pred, size, tp):
    if tp == 'whole':
        left_vec = crop_trace(field[i-neig], pred[i-neig][0], size)
        right_vec = crop_trace(field[i+neig], pred[i+neig][0], size)
        inv_left_vec = crop_trace((-1)*field[i-neig], pred[i-neig][1], size)
        inv_right_vec = crop_trace((-1)*field[i+neig], pred[i+neig][1], size)
        corr_left = np.corrcoef(left_vec, corr_vec)[0][1]
        corr_right = np.corrcoef(right_vec, corr_vec)[0][1]
        inv_corr_left = np.corrcoef(inv_left_vec, corr_vec)[0][1]
        inv_corr_right= np.corrcoef(inv_right_vec, corr_vec)[0][1]

    elif tp == 'on_left':
        corr_left = 0
        inv_corr_left = 0
        right_vec = crop_trace(field[i+neig], pred[i+neig][0], size)
        inv_right_vec = crop_trace((-1)*field[i+neig], pred[i+neig][1], size)
        corr_right = np.corrcoef(right_vec, corr_vec)[0][1]
        inv_corr_right = np.corrcoef(inv_right_vec, corr_vec)[0][1]

    elif tp == 'on_right':
        corr_right = 0
        inv_corr_right = 0
        left_vec = crop_trace(field[i-neig], pred[i-neig][0], size)
        inv_left_vec = crop_trace((-1)*field[i-neig], pred[i-neig][1], size)
        corr_left = np.corrcoef(left_vec, corr_vec)[0][1]
        inv_corr_left = np.corrcoef(inv_left_vec, corr_vec)[0][1]
    
    return corr_left, corr_right, inv_corr_left, inv_corr_right

class InverseBatch(SeismicBatch): #pylint: disable=too-few-public-methods
    """Class consists one action to generate a dataset
    with inverse traces. Depend on SeismicBatch class.
    """
    @action
    @inbatch_parallel(init='_init_component')
    def generate_inverse_dataset(self, index, model, num_neig, #pylint: disable=too-many-locals, too-many-arguments, too-many-statements
                                 src, mode='generate', dst=None, size=20):
        """Generate dataset using given model and field data.

        Parameters
        ----------
        model : torch model
            model to predict first break picking.
        num_neig : int
            number of neighbours to compare model results.
        src : str
            component's name with data
        mode : str
            There are two generation mode - 'generate' and 'predict'.
            If mode == 'generate' then function save dataset with equal number
            of normal and inverce traces to dst. If 'predict' then small number of traces
            will be inversed and dataset will have the same size as input field.
            If mode == 'without_inv' then dataset will consist the features for input traces.
        dst : str or None, optional
            component's name to save resulted data.
        size : int
            size of the vectors t
        """

        if mode not in ['generate', 'predict', 'without_inv']:
            raise ValueError('Incorrect value of "mode" parameter.')

        dst = src if dst is None else dst
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]
        offset = np.array(self.index.get_df(index=index)['offset'])

        pred = []
        for trace in field:
            tens_t = torch.Tensor(trace).reshape(1, 1, trace.shape[0]).to('cuda')
            inv_t = torch.Tensor(-trace).reshape(1, 1, trace.shape[0]).to('cuda')
            pred.append([predict(tens_t, model), predict(inv_t, model)])
        data = []

        inv_ix = np.zeros(field.shape[0])
        mask = np.random.choice(range(len(inv_ix)), size=np.random.randint(1, 5))
        inv_ix[mask] = 1
        
        lr = LinearRegression()
        data_tr = np.stack((offset, np.ones(len(offset)))).T
        target = np.array(pred)[:, 0].reshape(-1, 1)
        lr.fit(data_tr, target)
        off_pred = [int(lr.predict([[off, 1]])[0][0]) for off in offset]
        off_pred = np.array([off_pred, off_pred]).T
        
        for i, (trace, inv_trace, norm_predict, off) in enumerate(zip(field, -field,
                                                                      pred, offset)):
            traces = [(trace, norm_predict[0]), (inv_trace, norm_predict[1])]
            for j, (trs, prs) in enumerate(traces):

                if j == 1 and mode == 'predict' and inv_ix[i] != 1:
                    continue
                elif j == 0 and mode == 'predict' and inv_ix[i] == 1:
                    continue
                elif j == 1 and mode == 'without_inv':
                    continue
                amp_val = trs[prs]
                off_vec = crop_trace(trs, off_pred[i][0], size)
                corr_vec = crop_trace(trs, prs, size)

                corr_left = []
                corr_right = []
                inv_corr_left = []
                inv_corr_right = []
                of_corr_left = []
                of_corr_right = []
                of_inv_corr_left = []
                of_inv_corr_right = []
                for neig in range(1, num_neig+1):
                    if i >= neig and i < len(field) - neig:
                        result_corr = calculate_corr(corr_vec, field, i, neig, pred, size, 'whole')
                        corr_left.append(result_corr[0])
                        corr_right.append(result_corr[1])
                        inv_corr_left.append(result_corr[2])
                        inv_corr_right.append(result_corr[3])

                        off_result_corr = calculate_corr(off_vec, field, i, neig, off_pred, size, 'whole')
                        of_corr_left.append(off_result_corr[0])
                        of_corr_right.append(off_result_corr[1])
                        of_inv_corr_left.append(off_result_corr[2])
                        of_inv_corr_right.append(off_result_corr[3])

                    elif i < neig:
                        
                        result_corr = calculate_corr(corr_vec, field, i, neig, pred, size, 'on_left')
                        corr_left.append(result_corr[0])
                        corr_right.append(result_corr[1])
                        inv_corr_left.append(result_corr[2])
                        inv_corr_right.append(result_corr[3])
                        
                        off_result_corr = calculate_corr(off_vec, field, i, neig, off_pred, size, 'on_left')
                        of_corr_left.append(off_result_corr[0])
                        of_corr_right.append(off_result_corr[1])
                        of_inv_corr_left.append(off_result_corr[2])
                        of_inv_corr_right.append(off_result_corr[3])

                    else:
                        result_corr = calculate_corr(corr_vec, field, i, neig, pred, size, 'on_right')
                        corr_left.append(result_corr[0])
                        corr_right.append(result_corr[1])
                        inv_corr_left.append(result_corr[2])
                        inv_corr_right.append(result_corr[3])
                        
                        off_result_corr = calculate_corr(off_vec, field, i, neig, off_pred, size, 'on_right')
                        of_corr_left.append(off_result_corr[0])
                        of_corr_right.append(off_result_corr[1])
                        of_inv_corr_left.append(off_result_corr[2])
                        of_inv_corr_right.append(off_result_corr[3])
                data.append([amp_val, off, *of_corr_left, *of_corr_right,
                             *of_inv_corr_left, *of_inv_corr_right,
                             *corr_left, *corr_right,
                             *inv_corr_left, *inv_corr_right, j])
        getattr(self, dst)[pos] = data
        return self
