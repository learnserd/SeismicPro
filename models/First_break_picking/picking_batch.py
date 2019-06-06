import sys
import numpy as np

sys.path.append('../..')

from seismicpro.src import SeismicBatch
from seismicpro.src.seismic_batch import apply_to_each_component
from seismicpro.batchflow import action, inbatch_parallel
from seismicpro.batchflow.batch_image import transform_actions


@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class PickingBatch(SeismicBatch):
    
    @action
    def new_mask(self, src, dst):
        """ Convert picking time to the mask.
        """
        m = np.concatenate(np.vstack(getattr(self, src))).copy()
        m = np.around(m / 2).astype('int')
        bs = len(getattr(self, src))
        length = self.raw[0].shape[1]
        ind = tuple(np.array(list(zip(range(bs), m))).T)
        
        mask = np.zeros((bs, length))
        mask[ind] = 1
        setattr(self, dst, np.cumsum(mask, axis=1))
        return self
        
    @action
    def normalize_traces(self, src, dst):
        """ Normalize traces to zero mean and unit variance.
        """  
        data = getattr(self, src)
        data = np.vstack(data)
        res = (data - np.mean(data, axis=1)[:,np.newaxis]) / np.std(data, axis=1)[:, np.newaxis]
        res = res.reshape(res.shape[0], 1, -1)
        setattr(self, dst, res)
        return self
    
    @action
    def mask_to_pick(self, src, dst, labels=True):
        """ Convert the mask to picking time.
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
    @inbatch_parallel(init="_init_component", target="threads")
    def normalize_traces_parallel(self, index, src, dst):
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        data = (data - data.mean()) / data.std()
        getattr(self, dst)[pos] = data
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    def get_mask_parallel(self, index, src, dst):
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        time = int(round(data[0]/2))
        trace_len = getattr(self, 'raw')[pos].shape[1]
        mask = np.zeros(trace_len)
        mask[time:] = 1
        getattr(self, dst)[pos] = mask
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads", post='_post_mask_to_sample')
    def mask_to_sample(self, index, src, dst, **kwargs):
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        return mass_ones(data, **kwargs)

    def _post_mask_to_sample(self, picking, *args, **kwargs):
        comp = kwargs['dst']
        setattr(self, comp, np.array(picking))
        return self
    
        