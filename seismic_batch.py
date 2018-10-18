import glob
import os
import numpy as np
import segyio
import matplotlib.pyplot as plt

import dataset as ds

def make_data(slice_data, half_size):
    hs = half_size    
    slice_data_ext = np.pad(slice_data, ((3*half_size, 3*half_size - 1),
                                         (3*half_size, 3*half_size - 1)),
                            'symmetric')
    d_coord = np.zeros_like(slice_data_ext) + np.arange(len(slice_data_ext[0])) - 3*half_size
    x_coord = np.zeros_like(slice_data_ext) + np.arange(len(slice_data_ext)).reshape((-1, 1)) - 3*half_size
    
    all_stacked = np.stack([slice_data_ext[:-4*hs, 2*hs:-2*hs],
                            slice_data_ext[2*hs:-2*hs, 2*hs:-2*hs],
                            slice_data_ext[4*hs:, 2*hs:-2*hs:],
                            slice_data_ext[2*hs:-2*hs, :-4*hs],
                            slice_data_ext[2*hs:-2*hs, 2*hs:-2*hs],
                            slice_data_ext[2*hs:-2*hs, 4*hs:],
                            x_coord[2*hs:-2*hs, 2*hs:-2*hs],
                            d_coord[2*hs:-2*hs, 2*hs:-2*hs]], axis=-1)    
    return all_stacked

def rolling_window(a, shape):  # rolling window along axis=(0, 1) for >= 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape + a.shape[2:]
    strides = a.strides[:2] + a.strides[:2] + a.strides[2:]
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

def pts_to_indices(pts, meta):
    starts = np.array([meta.ilines[0], meta.xlines[0], meta.samples[0]])
    steps = np.array([meta.ilines[1] - meta.ilines[0],
                      meta.xlines[1] - meta.xlines[0],
                      meta.samples[1] - meta.samples[0]])
    return (pts - starts) / steps

def get_grid_points(shape, stride):
    ind = np.transpose(np.indices(shape), tuple(np.arange(1, len(shape) + 1)) + (0,))
    return (ind[[slice(None, None, stride)] * len(shape) + [slice(None)]]
            .reshape((-1, len(shape))))

class SeismicBatch(ds.Batch):
    def __init__(self, index, preloaded=None):
        components = "traces", "annotation", "meta"
        super().__init__(index, preloaded=preloaded)
        if preloaded is None:
            self.traces = np.array([None] * len(self.index))
            self.annotation = np.array([None] * len(self.index))
            self.meta = np.array([None] * len(self.index))

    @ds.action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        return self._load_data(src, fmt, components)
            
    @ds.inbatch_parallel(init="indices", target="threads")
    def _load_data(self, index, src=None, fmt=None, components=None, *args, **kwargs):
        if src is not None:
            path = src[index]
        if isinstance(self.index, ds.FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")
        pos = self.get_pos(None, "indices", index)
        if fmt == "segy":
            self.traces[pos] = segyio.tools.cube(path)
            self.meta[pos] = segyio.open(path)
        elif fmt == "pts":
            pdir = os.path.split(path)[0] + '/*.pts'
            files = glob.glob(pdir)
            self.annotation[pos] = []
            for f in files:
                self.annotation[pos].append(np.loadtxt(f))
        else:
            raise NotImplementedError("Unknown file format.")
    
    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list.
        """
        if ds.any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)
    
    def _assemble_crops(self, results, *args, **kwargs):
        """Concatenate results of different workers.
        """
        _ = args, kwargs
        self._reraise_exceptions(results)
        crops, labels = list(zip(*results))
                
        crops = np.vstack(crops)
        labels = np.hstack(labels)
            
        return ds.ImagesBatch(ds.DatasetIndex(np.arange(len(crops))),
                              preloaded=(crops, labels, np.zeros(len(crops))))
    
    @ds.action
    @ds.inbatch_parallel(init="indices", post="_assemble_crops", target="threads")
    def sample_crops(self, index, axis, offset, crop_half_size, n_crops):
        hs = crop_half_size
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]
        ix = [slice(None)] * traces.ndim
        ix[axis] = offset
        
        slice_data = make_data(traces[ix], hs)
        
        weights = np.array([len(arr) for arr in pts])
        weights = weights / weights.sum()
        
        unique_labels = np.arange(len(pts))
        labels = np.random.choice(unique_labels, p=weights, size=n_crops)
        
        ipts = np.array([pts_to_indices(p[:, :3], meta) for p in pts])
        
        crops = []
        
        for i in labels:
            p = ipts[i][np.random.randint(len(pts[i]))].astype(int)
            p = np.delete(p, axis)
            crop = slice_data[p[0]:p[0] + 2*hs, p[1]:p[1] + 2*hs]
            crops.append(crop)
        crops = np.array(crops)
        return [crops, labels]
    
    @ds.action
    @ds.inbatch_parallel(init="indices", post="_assemble_crops", target="threads")
    def next_crops(self, index, slice_axis, slice_index,
                   n_crops, crop_half_size, stride, start_index):
        hs = crop_half_size
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]      
        
        ix = [slice(None)] * traces.ndim
        ix[slice_axis] = slice_index
        
        slice_data = make_data(traces[ix], hs)
        
        crops = []
        grid_pts = get_grid_points(traces[ix].shape, stride)
                
        for p in grid_pts[start_index: start_index + n_crops]:
            crop = slice_data[p[0]:p[0] + 2*hs, p[1]:p[1] + 2*hs]
            crops.append(crop)
                
        crops = np.array(crops)
        return [crops, np.array([None] * len(crops))]
    
    def show_slice(self, index, axis, offset, show_pts=False):
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]
        ix = [slice(None)] * traces.ndim
        ix[axis] = offset       
        slice_data = traces[ix]
        axes_names = np.delete(["i-lines", "x-lines", "depth"], axis)
        
        plt.imshow(traces[ix].T, cmap="gray")
        
        if show_pts:
            ax = np.delete(np.arange(3), axis)
            ipts = np.array([pts_to_indices(p[:, :3], meta) for p in pts])
            for arr in ipts:
                plt.scatter(arr[:, ax[0]], arr[:, ax[1]], alpha=0.005)

        plt.ylabel(axes_names[1])
        plt.xlabel(axes_names[0])
        plt.show()
