import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')

from seismicpro.src import SeismicBatch
from seismicpro.src.seismic_batch import apply_to_each_component
from seismicpro.batchflow import action, inbatch_parallel
from seismicpro.batchflow import Pipeline, Dataset, B, V, F
from seismicpro.batchflow.batch_image import transform_actions
from seismicpro.batchflow.models.metrics import Metrics

def test_pipeline(train_pipeline):
    test_pipeline = (Pipeline()
                          .import_model('my_model', train_pipeline) 
                          .load(components='raw', fmt='segy')
                          .load(components='markup', fmt='picks')
                          .drop_zero_traces(num_zero=700, src='raw')
                          .normalize_traces(src='raw', dst='raw')
                          .mask_to_picking(src='markup', dst='mask')
                          .init_variable('picking', init_on_each_run=list())
                          .update_variable('picking', B('mask',))
                          .add_components(components='unet_predictions')
                          .predict_model('my_model', B('raw'), fetches=['predictions'],
                                          save_to=[B('unet_predictions')], mode='a')
                          .init_variable('traces', init_on_each_run=list())
                          .update_variable('traces', B('raw'), mode='a')
                          .mask_to_pick(src='unet_predictions', dst='unet_predictions', labels=False)
                          .init_variable('predictions', init_on_each_run=list())
                          .update_variable('predictions', B('unet_predictions'), mode='a'))
    return test_pipeline

class PickingMetrics(Metrics):
    def __init__(self, targets, predictions, diff=3):
        super().__init__()
        self.targets=targets
        self.predictions=predictions
        self.diff = diff

    def MAE(self, *args, **kwargs):
        return np.mean(np.abs(self.targets - self.predictions))
    
    def accuracy(self, *args, **kwargs):
        d = np.abs(self.targets - self.predictions)
        return 100* len(d[d < self.diff]) / len(d)

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class PickingBatch(SeismicBatch):
    @action
    def mask_to_picking(self, src, dst):
        """Convert picking time to the mask.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.

        Returns
        -------
        batch : SeismicBatch
            Batch with the mask corresponds to the picking.
        """
        m = np.concatenate(np.vstack(getattr(self, src)))
        m = np.around(m / 2).astype('int')
        bs = len(getattr(self, src))
        length = self.raw[0].shape[1]
        ind = tuple(np.array(list(zip(range(bs), m))).T)
        ind[1][ind[1] < 0] = 0
        mask = np.zeros((bs, length))
        mask[ind] = 1
        setattr(self, dst, np.cumsum(mask, axis=1))
        return self
        
    @action
    def normalize_traces(self, src, dst):
        """Normalize traces to zero mean and unit variance.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.

        Returns
        -------
        batch : SeismicBatch
            Batch with the normalized traces.
        """  
        data = getattr(self, src)
        data = np.vstack(data)
        res = (data - np.mean(data, axis=1)[:, np.newaxis]) / np.std(data, axis=1)[:, np.newaxis]
        res = res.reshape(res.shape[0], 1, -1)
        setattr(self, dst, res)
        return self
    
    @action
    def mask_to_pick(self, src, dst, labels=True):
        """Convert the mask to picking time.
        
        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        labels: bool, default: False
            The flag indicates whether action's inputs probabilities or labels.

        Returns
        -------
        batch : SeismicBatch
            Batch with the predicted picking times.
        """  
    
        data = getattr(self, src)
        if not labels:
            data = np.argmax(data, axis=1)
        arr = np.append(data, np.zeros((data.shape[0], 1)), axis=1)
        arr = np.insert(arr, 0, 0, axis=1)

        plus_one = np.argwhere((np.diff(arr)) == 1)
        minus_one = np.argwhere((np.diff(arr)) == -1)

        d = minus_one[:, 1] - plus_one[:, 1]
        mask = minus_one[:, 0]

        sort = np.lexsort((d, mask))
        ind = [0] * mask[0]
        for i in range(len(sort[:-1])):
            diff = mask[i +1] - mask[i]
            if diff > 1:
                ind.append(plus_one[:, 1][sort[i]])
                ind.extend([0] * (diff - 1))
            elif diff == 1:
                ind.append(plus_one[:, 1][sort[i]])
        ind.append(plus_one[:, 1][sort[-1]])
        ind.extend([0] * (arr.shape[0] - mask[-1] - 1))
        ind = [[i] for i in ind]
        setattr(self, dst, ind)
        return self
    
    @action
    def MCM(self, src, dst, eps=3, l=12):
        """Creates for each trace corresponding Energy function.
        Based on Coppens(1985) method.
        
        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        eps: float, default: 3
            Stabilization constant that helps reduce the rapid fluctuations of energy function.
        l: int, default: 12
            The leading window length.

        Returns
        -------
        batch : SeismicBatch
            Batch with the energy function.
        """ 
        trace = np.concatenate(getattr(self, src))
        energy = np.cumsum(trace**2, axis=1)
        long_win, lead_win = energy, energy
        lead_win[:,l:] = lead_win[:, l:] - lead_win[:, :-l]
        er = lead_win / (long_win + eps)
        self.add_components(dst, init=er)
        return self
    
    @action   
    def energy_to_picking(self, src, dst):
        """Convert energy function of the trace to the picking time by taking derivative 
        and finding maximum.
        
        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.

        Returns
        -------
        batch : SeismicBatch
            Batch with the predicted picking by MCM method.
        """
        er = getattr(self, src)
        er = np.gradient(er, axis=1)
        picking = np.argmax(er, axis=1)
        self.add_components(dst, init=picking)
        return self
     
def seismic_plot(arrs, wiggle=False, xlim=None, ylim=None, std=1, # pylint: disable=too-many-branches, too-many-arguments
                 pts=None, s=None, c=None, names=None, figsize=None,
                 save_to=None, dpi=None, **kwargs):
    """Plot seismic traces.

    Parameters
    ----------
    arrs : array-like
        Arrays of seismic traces to plot.
    wiggle : bool, default to False
        Show traces in a wiggle form.
    xlim : tuple, optional
        Range in x-axis to show.
    ylim : tuple, optional
        Range in y-axis to show.
    std : scalar, optional
        Amplitude scale for traces in wiggle form.
    pts : array_like, shape (n, )
        The points data positions.
    s : scalar or array_like, shape (n, ), optional
        The marker size in points**2.
    c : color, sequence, or sequence of color, optional
        The marker color.
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    save_to : str or None, optional
        If not None, save plot to given path.
    dpi : int, optional, default: None
        The resolution argument for matplotlib.pyplot.savefig.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Multi-column subplots.
    """
    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    if isinstance(names, str):
        names = (names,)

    _, ax = plt.subplots(1, len(arrs), figsize=figsize, squeeze=False)
    for i, arr in enumerate(arrs):
        if not wiggle:
            arr = np.squeeze(arr)

        if xlim is None:
            xlim = (0, len(arr))

        if arr.ndim == 2:
            if ylim is None:
                ylim = (0, len(arr[0]))

            if wiggle:
                offsets = np.arange(*xlim)
                y = np.arange(*ylim)
                for k in offsets:
                    x = k + std * arr[k, slice(*ylim)] / np.std(arr)
                    ax[0, i].plot(x, y, 'k-')
                    ax[0, i].fill_betweenx(y, k, x, where=(x > k), color='k')

            else:
                ax[0, i].imshow(arr.T, **kwargs)

        elif arr.ndim == 1:
            ax[0, i].plot(arr, **kwargs)
        else:
            raise ValueError('Invalid ndim to plot data.')

        if pts is not None:
            if isinstance(pts, list):
                if not wiggle:
                    ax[0, i].scatter(*pts[0], s=s, c=c[0], alpha = 1)
                    ax[0, i].scatter(*pts[1], s=s, c=c[1], alpha = 0.4)
                    #plt.legend(['Predictions','Labels'], loc='upper right')
                else:
                    ax[0, i].scatter(*pts[0], s=s, c=c[0], alpha = 1)
                    ax[0, i].scatter(*pts[1], s=s, c=c[1], alpha = 0.6)
            
            else:
                ax[0, i].scatter(*pts, s=s, c=c)

        if names is not None:
            ax[0, i].set_title(names[i])

        if arr.ndim == 2:
            ax[0, i].set_ylim([ylim[1], ylim[0]])
            if (not wiggle) or (pts is not None):
                ax[0, i].set_xlim(xlim)

        if arr.ndim == 1:
            plt.xlim(xlim)

        ax[0, i].set_aspect('auto')

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)

    plt.show()