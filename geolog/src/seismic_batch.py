"""Seismic batch."""
import glob
import os
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import segyio

from batchflow import (action, inbatch_parallel, Batch,
                       DatasetIndex,
                       ImagesBatch, any_action_failed)

from .field_index import SegyFilesIndex, TraceIndex, DataFrameIndex
from .utils import IndexTracker, partialmethod
from .batch_tools import nj_sample_crops, pts_to_indices


ACTIONS_DICT = {
    "clip": (np.clip, "numpy.clip", "clip values"),
    "gradient": (np.gradient, "numpy.gradient", "gradient"),
    "fft2": (np.fft.fft2, "numpy.fft.fft2", "a Discrete 2D Fourier Transform"),
    "ifft2": (np.fft.ifft2, "numpy.fft.ifft2", "an inverse Discrete 2D Fourier Transform"),
    "fft": (np.fft.fft, "numpy.fft.fft", "a Discrete Fourier Transform"),
    "ifft": (np.fft.ifft, "numpy.fft.ifft", "an inverse Discrete Fourier Transform"),
    "rfft": (np.fft.rfft, "numpy.fft.rfft", "a real-input Discrete Fourier Transform"),
    "irfft": (np.fft.irfft, "numpy.fft.irfft", "a real-input inverse Discrete Fourier Transform"),
    "dwt": (pywt.dwt, "pywt.dwt", "a single level Discrete Wavelet Transform"),
    "idwt": (lambda x, *args, **kwargs: pywt.idwt(*x, *args, **kwargs), "pywt.idwt",
             "a single level inverse Discrete Wavelet Transform"),
    "wavedec": (pywt.wavedec, "pywt.wavedec", "a multilevel 1D Discrete Wavelet Transform"),
    "waverec": (lambda x, *args, **kwargs: pywt.waverec(list(x), *args, **kwargs), "pywt.waverec",
                "a multilevel 1D Inverse Discrete Wavelet Transform"),
    "pdwt": (lambda x, part, *args, **kwargs: pywt.downcoef(part, x, *args, **kwargs), "pywt.downcoef",
             "a partial Discrete Wavelet Transform data decomposition"),
    "cwt": (lambda x, *args, **kwargs: pywt.cwt(x, *args, **kwargs)[0].T, "pywt.cwt", "a Continuous Wavelet Transform"),
}


TEMPLATE_DOCSTRING = """
    TBD.
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()


def add_actions(actions_dict, template_docstring):
    """Add new actions
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            docstring = template_docstring.format(full_name=full_name, description=description)
            method = partialmethod(cls.apply_to_each_channel, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)
        return cls
    return decorator

@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)  # pylint: disable=too-many-public-methods,too-many-instance-attributes
class SeismicBatch(Batch):
    """Docstring."""
    components = 'traces', 'annotation', 'meta'
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded=preloaded)
        if preloaded is None:
            self.traces = np.array([None] * len(self.index))
            self.annotation = np.array([None] * len(self.index))
            self.meta = np.array([dict()] * len(self.index))

    def _init_component(self, *args, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        return self.indices

    @action
    @inbatch_parallel(init="_init_component", src="traces", dst="traces", target="threads")
    def apply_to_each_channel(self, index, func, *args, src="traces", dst="traces", **kwargs):
        """TBD.
        """
        i = self.get_pos(None, src, index)
        src_data = getattr(self, src)[i]
        dst_data = np.array([func(slc, *args, **kwargs) for slc in src_data])
        getattr(self, dst)[i] = dst_data

    @action
    def apply_transform(self, func, *args, src="traces", dst="traces", **kwargs):
        """Docstring."""
        super().apply_transform(func, *args, src=src, dst=dst, **kwargs)
        dst_data = getattr(self, dst)
        setattr(self, dst, np.array([i for i in dst_data] + [None])[:-1])
        return self

    @action
    @inbatch_parallel(init="_init_component", src="traces", dst="traces", target="threads")
    def shift_traces(self, index, src="traces", shift_src=None, dst="traces"):
        """TBD.
        """
        i = self.get_pos(None, src, index)
        traces = getattr(self, src)[i]
        if isinstance(shift_src, str):
            shifts = getattr(self, shift_src)[i]

        dst_data = np.array([traces[k][max(0, shifts[k]):] for k in range(len(traces))])
        getattr(self, dst)[i] = dst_data

    @action
    @inbatch_parallel(init="_init_component", src="traces", dst="traces", target="threads")
    def band_pass_filter(self, index, lowcut=None, highcut=None, fs=1, order=5, src="traces", dst="traces"):
        """TBD.
        """
        i = self.get_pos(None, src, index)
        traces = getattr(self, src)[i]
        nyq = 0.5 * fs
        if lowcut is None:
            b, a = signal.butter(order, highcut / nyq, btype='high')
        elif highcut is None:
            b, a = signal.butter(order, lowcut / nyq, btype='low')
        else:
            b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        getattr(self, dst)[i] = signal.lfilter(b, a, traces)

    @action
    @inbatch_parallel(init="indices", target="threads")
    def to_2d(self, index, component='traces', length_alingment=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces = getattr(self, component)[pos]
        if traces is None or len(traces) == 0:
            return
        try:
            traces_2d = np.vstack(traces)
        except ValueError as err:
            if length_alingment is None:
                raise ValueError(str(err) + '\nTry to set length_alingment to \'max\' or \'min\'')
            elif length_alingment == 'min':
                nsamples = min([len(t) for t in traces])
            elif length_alingment == 'max':
                nsamples = max([len(t) for t in traces])
            else:
                raise NotImplementedError('Unknown length_alingment')
            shape = (len(traces), nsamples)
            traces_2d = np.zeros(shape)
            for i, arr in enumerate(traces):
                traces_2d[i, :len(arr)] = arr[:nsamples]
        getattr(self, component)[pos] = traces_2d

    @action
    def stack(self, components):
        """Docstring."""
        res = type(self)(DatasetIndex(1))
        for component in components:
            data = getattr(self, component)
            setattr(res, component, np.array([np.vstack(data)]))
        res.meta[0] = dict(sorting=None)
        return res

    @action
    def dump(self, path, fmt, component='traces', **kwargs):
        """Docstring."""
        if fmt in ['sgy', 'segy']:
            self._dump_segy(path, component, **kwargs)
        else:
            raise NotImplementedError('Unknown file format.')

    @inbatch_parallel(init="indices", target="threads")
    def _dump_segy(self, index, path, component, **kwargs):
        data = getattr(self, component)[index]
        path = os.path.join(path, str(index) + '.sgy')
        if data.ndim == 1:
            segyio.tools.from_array(path, data=data, **kwargs)
        elif data.ndim == 2:
            segyio.tools.from_array2D(path, data=data, **kwargs)
        elif data.ndim == 3:
            segyio.tools.from_array3D(path, data=data, **kwargs)
        elif data.ndim == 4:
            segyio.tools.from_array4D(path, data=data, **kwargs)
        else:
            raise ValueError('Invalid data ndim.')

    @action
    def load(self, src=None, fmt=None, components=None, **kwargs):
        """Docstring."""
        if isinstance(self.index, DataFrameIndex):
            return self._load_segy(components=components, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    def _load_segy(self, components='traces', sort_by='trace_number'):
        """Docstring."""
        idf = self.index._idf # pylint: disable=protected-access

        if isinstance(components, str):
            components = (components,)

        trace_index = TraceIndex(self.index).ravel(name='traces', order=components)
        order = []
        for i, group in trace_index._idf.groupby(by=('traces', 'file_id')): # pylint: disable=protected-access
            order.extend(group.index.tolist())

        segy_index = SegyFilesIndex(trace_index, name='traces')
        idf2 = segy_index._idf # pylint: disable=protected-access

        batch = type(self)(segy_index)._load_from_segy_files() # pylint: disable=protected-access
        all_traces = np.array([t for item in batch.traces for t in item] + [None])[:-1]
        idf2['_trace'] = None
        idf2.iloc[order, idf2.columns.get_loc('_trace')] = all_traces

        comp_values = np.split(idf2['_trace'].values, len(components))
        for i, comp in enumerate(components):
            idf['_' + comp] = comp_values[i]
            setattr(self, comp, np.array([None] * len(self)))

        if isinstance(self.index, TraceIndex):
            pos = [self.get_pos(None, "indices", i) for i in self.indices]
            for comp in components:
                getattr(self, comp)[pos] = np.array(idf['_' + comp].tolist() + [None])[:-1]
        else:
            for i in self.indices:
                ipos = self.get_pos(None, "indices", i)
                df = idf.loc[[i]].reset_index().sort_values(by=sort_by)
                for comp in components:
                    getattr(self, comp)[ipos] = df['_' + comp].values
                self.meta[ipos].update(dict(sorting=sort_by))

        return self

    @inbatch_parallel(init="indices", target="threads")
    def _load_from_segy_files(self, index, component='traces'):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        path = index
        idf = self.index._idf.loc[index][component] # pylint: disable=protected-access
        with segyio.open(path, strict=False) as segyfile:
            traces = np.array([segyfile.trace[i] for i in np.atleast_1d(idf['seq_number'])] + [None])[:-1]

        self.traces[pos] = traces
        self.meta[pos] = dict(sorting=None)
        return self

    @action
    @inbatch_parallel(init="indices", target="threads")
    def sort_traces(self, index, sort_by):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        if isinstance(self.index, DataFrameIndex):
            idf = self.index._idf.loc[index] # pylint: disable=protected-access
        else:
            raise ValueError("Sorting is not supported for this Index class")
        order = np.argsort(idf[sort_by].values)
        self.traces[pos] = self.traces[pos][order]
        self.meta[pos]['sorting'] = sort_by

    @action
    @inbatch_parallel(init="indices", target="threads")
    def summarize(self, index, axis=0, keepdims=True, max_offset=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        if max_offset is not None:
            sort_by = self.meta[pos]['sorting']
            offset = np.sort(self.index._idf.loc[index, sort_by].values) # pylint: disable=protected-access
            mask = np.where(offset < max_offset ** 2)[0]
        else:
            mask = slice(0, None, None)
        self.traces[pos] = np.mean(self.traces[pos][mask], axis=axis, keepdims=keepdims)

    @action
    @inbatch_parallel(init="indices", target="threads")
    def filter_annotations(self, index, ann_names, mode):
        """Docstring."""
        if isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")
        pdir = os.path.split(path)[0] + '/*.pts'
        files = np.array([os.path.split(p)[1] for p in glob.glob(pdir)])
        pos = self.get_pos(None, "indices", index)
        indices = []
        for ann in ann_names:
            i = np.where(files == ann)[0]
            if len(i):
                indices.append(i[0])
        if mode == "drop":
            self.annotation[pos] = np.delete(self.annotation[pos], indices)
        elif mode == "keep":
            self.annotation[pos] = self.annotation[pos][indices]
        else:
            raise ValueError("Unknown filter mode")
        return self

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list.
        """
        if any_action_failed(results):
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

        return ImagesBatch(DatasetIndex(np.arange(len(crops))),
                           preloaded=(crops, labels, np.zeros(len(crops))))

    @action
    @inbatch_parallel(init="indices", post="_assemble_crops", target="threads")
    def sample_crops(self, index, size, origin, n_crops=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]

        if isinstance(origin, (list, tuple, np.ndarray)):
            labels = np.array([None] * len(origin))
            sampled_pts = np.array(origin)

        elif origin == "random_annotated_unbalanced":
            stacked_pts = np.vstack(pts)[:, :3]
            pos = np.random.randint(0, len(stacked_pts), size=n_crops)
            sampled_pts = pts_to_indices(stacked_pts[pos], meta)
            labels = np.repeat(np.arange(len(pts)), [len(p) for p in pts])[pos]

        elif origin == "random_annotated_balanced":
            labels = np.random.choice(np.arange(len(pts)), size=n_crops)
            sampled_pts = np.zeros((n_crops, 3))
            for i, arr in enumerate(pts):
                mask = np.where(labels == i)
                pos = np.random.randint(0, len(arr), size=len(mask[0]))
                sampled_pts[mask] = arr[pos, :3]
            sampled_pts = pts_to_indices(sampled_pts, meta)
        else:
            raise ValueError("Unknown sampling mode")

        if isinstance(size, int):
            size = tuple([size] * traces.ndim)

        if traces.ndim == 2:
            size = size + (1,)

        crops = (nj_sample_crops(np.atleast_3d(traces), sampled_pts, size)
                 .reshape((-1,) + size[:traces.ndim]))

        return [crops, labels]

    def slice_tracker(self, index, axis, scroll_step=1, show_pts=False, **kwargs):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]

        order = np.hstack((np.delete(np.arange(traces.ndim), axis)[::-1], axis))
        axes_names = np.delete(["i-lines", "x-lines", "samples"], axis)

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(axes_names[0])
        ax.set_ylabel(axes_names[1])

        if show_pts:
            ipts = [pts_to_indices(arr[:, :3], meta)[:, order] for arr in pts]
        else:
            ipts = None
        tracker = IndexTracker(ax, np.transpose(traces, order), scroll_step=scroll_step,
                               pts=ipts, axes_names=axes_names, **kwargs)
        return fig, tracker

    def show_slice(self, index, axis=-1, offset=0, show_pts=False,
                   figsize=None, save_to=None, dpi=None, component='traces', **kwargs):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces = np.atleast_3d(getattr(self, component)[pos])
        pts, meta = self.annotation[pos], self.meta[pos]
        ix = [slice(None)] * traces.ndim
        ix[axis] = offset
        ax = np.delete(np.arange(3), axis)

        if meta["sorting"] == 2:
            axes_names = np.delete(["i-lines", "x-lines", "samples"], axis)
        elif meta["sorting"] == 1:
            axes_names = np.delete(["x-lines", "i-lines", "samples"], axis)
        else:
            axes_names = np.delete(["x", "y", "z"], axis)

        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.imshow(traces[ix].T, **kwargs)

        if show_pts:
            ipts = np.array([pts_to_indices(p[:, :3], meta) for p in pts])
            for arr in ipts:
                arr = arr[arr[:, axis] == offset]
                plt.scatter(arr[:, ax[0]], arr[:, ax[1]], alpha=0.005)

        plt.ylabel(axes_names[1])
        plt.xlabel(axes_names[0])
        plt.axis('auto')
        plt.xlim([0, traces.shape[ax[0]]])
        plt.ylim([traces.shape[ax[1]], 0])
        if save_to is not None:
            plt.savefig(save_to, dpi=dpi)
        plt.show()
