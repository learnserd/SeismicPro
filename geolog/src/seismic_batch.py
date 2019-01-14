"""Seismic batch."""
import glob
import os
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import segyio

from batchflow import (action, inbatch_parallel, Batch,
                       FilesIndex, DatasetIndex,
                       ImagesBatch, any_action_failed)

from .field_index import FieldIndex
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
            self.meta = np.array([None] * len(self.index))

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
    def to_2d(self, index, length_alingment=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces = self.traces[pos]
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
        self.traces[pos] = traces_2d

    @action
    def stack_traces(self, component='traces'):
        """Docstring."""
        res = type(self)(DatasetIndex(1))
        data = getattr(self, component)
        res.traces[0] = np.vstack(data)
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
    def load(self, src=None, path=None, fmt=None, components=None, *args, **kwargs):
        """Docstring."""
        if isinstance(self.index, FilesIndex) or (src is not None and fmt is not None):
            return self._load_from_paths(src=src, fmt=fmt, *args, **kwargs)
        if isinstance(self.index, FieldIndex):
            return self._load_from_traces(path=path, fmt=fmt, *args, **kwargs)
        if isinstance(src, pd.DataFrame):
            return self._load_from_df(src, *args, **kwargs)
        if path is not None:
            return self._load_from_one_path(path, fmt=fmt, components=components, *args, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, *args, **kwargs)

    @action
    def _load_from_df(self, src, force=False):
        """Docstring."""
        df = src.loc[self.indices]
        for component in df.columns:
            if force:
                setattr(self, component, df[component].values)
            elif hasattr(self, component) :
                setattr(self, component, df[component].values)
        return self

    def _load_from_one_path(self, path, fmt, components='traces', skip_channels=0):
        """Docstring."""
        if fmt == "segy":
            with segyio.open(path, strict=False) as file:
                traces = np.array([np.atleast_2d(file.trace[i])
                                   for i in self.indices - 1 + skip_channels] + [None])[:-1]
            setattr(self, components, traces)
            return self
        else:
            raise NotImplementedError("Unknown file format.")

    def _load_from_traces(self, path=None, fmt=None, sort_by='r2',
                          get_file_by_index=None, skip_channels=0):
        """Docstring."""
        src = []
        channels = []
        pos = []

        idf = self.index._idf # pylint: disable=protected-access
        idf['_pos'] = np.arange(len(idf))

        for index, group in idf.groupby(['tape', 'xid']):
            file = get_file_by_index(path, index)
            if file is not None:
                src.append(file)
                channels.append(group['channel'].values)
                pos.extend(group['_pos'].values)

        if not src:
            return self

        batch = (type(self)(DatasetIndex(np.arange(len(src))))
                 .load(src=src, fmt=fmt, channels=channels, skip_channels=skip_channels))

        all_traces = np.array([t for item in batch.traces for t in item] + [None])[:-1]

        idf['_trace'] = np.nan
        idf.iloc[pos, idf.columns.get_loc('_trace')] = all_traces

        for index, group in idf.groupby(level=0):
            ipos = self.get_pos(None, "indices", index)
            group = group.dropna(axis=0).sort_values(by=sort_by)
            self.traces[ipos] = group['_trace'].values
            self.meta[ipos] = dict(sorting=sort_by,
                                   sht_depth=group['sht_depth'].values if 'sht_depth' in group.columns else None,
                                   uphole=group['uphole'].values if 'uphole' in group.columns else None,
                                   z=group['z_s'].values if 'z_s' in group.columns else None)

        idf.drop(labels=['_pos', '_trace'], axis=1, inplace=True)

        return self


    @inbatch_parallel(init="indices", target="threads")
    def _load_from_paths(self, index, src=None, fmt=None, channels=None, skip_channels=0):
        """Docstring."""
        if src is not None:
            path = src[index]
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source is not specified")
        pos = self.get_pos(None, "indices", index)
        if fmt == "segy":
            with segyio.open(path, strict=False) as file:
                if (file.sorting is not None) and (channels is None):
                    self.traces[pos] = segyio.tools.cube(file)
                else:
                    if channels is None:
                        self.traces[pos] = file.trace.raw[skip_channels:]
                    else:
                        self.traces[pos] = np.array([file.trace[i] for i in channels[index] - 1 + skip_channels])
                self.meta[pos] = segyio.tools.metadata(file).__dict__
        elif fmt == "pts":
            pdir = os.path.split(path)[0] + '/*.pts'
            files = glob.glob(pdir)
            self.annotation[pos] = []
            for file in files:
                self.annotation[pos].append(np.loadtxt(file))
            self.annotation[pos] = np.array(self.annotation[pos])
        else:
            raise NotImplementedError("Unknown file format.")

    @action
    @inbatch_parallel(init="indices", target="threads")
    def sort_traces(self, index, sort_by):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        if isinstance(self.index, FieldIndex):
            idf = self.index._idf.loc[index] # pylint: disable=protected-access
        else:
            raise ValueError("Sorting is not supported for this Index class")
        order = np.argsort(idf[sort_by].values)
        self.traces[pos] = self.traces[pos][order]
        self.meta[pos]['sorting'] = sort_by

    @action
    @inbatch_parallel(init="indices", target="threads")
    def summarize(self, index, axis=0, keepdims=True, max_r=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        if max_r is not None:
            sort_by = self.meta[pos]['sorting']
            r2 = np.sort(self.index._idf.loc[index, sort_by].values)
            mask = np.where(r2 < max_r ** 2)[0]
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

    def show_slice(self, index, axis=-1, offset=0, show_pts=False, figsize=None, save_to=None, dpi=None, **kwargs):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = np.atleast_3d(self.traces[pos]), self.annotation[pos], self.meta[pos]
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
