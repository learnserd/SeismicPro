import sys
import numpy as np

sys.path.append('../..')

from seismicpro.src import SeismicBatch
from seismicpro.src.seismic_batch import apply_to_each_component
from seismicpro.batchflow import action, inbatch_parallel
from seismicpro.batchflow.batch_image import transform_actions

@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class my_batch(SeismicBatch):
    @action
    @inbatch_parallel(init="_init_component", target="threads")
    def get_mask(self, index, src, dst):
        pos = self.get_pos(None, src, index)
        try:
            data = getattr(self, src)[pos]
        except:
            print(pos, len(getattr(self, src)))
        time = int(round(data[0]/2))
        trace_len = getattr(self, 'raw')[pos].shape[1]
        mask = np.zeros(trace_len)
        mask[time:] = 1
        getattr(self, dst)[pos] = mask
        return self

    @action
   # @apply_to_each_component
    def process_component(self, src, dst, add_dim=False):
        data = getattr(self, src)
        data = np.vstack(data).astype(np.float32)
        if add_dim:
            data = np.expand_dims(data, axis=1)
        if dst in self.components:
            setattr(self, dst, data)
        else:
            self.add_components(dst, init=data)
        return self
    
    
    @action
    @inbatch_parallel(init="_init_component", target="threads")
    def normalize_traces(self, index, src, dst):
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        data = (data - data.mean()) / data.std()
        getattr(self, dst)[pos] = data
        return self
    
    