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
                       DatasetIndex,
                       ImagesBatch, any_action_failed)

from .field_index import SegyFilesIndex, TraceIndex, DataFrameIndex, FILE_DEPENDEND_COLUMNS, DEFAULT_SEGY_HEADERS
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

def apply_to_each_component(method):
    """Docstring."""
    def decorator(self, *args, **kwargs):
        """Docstring."""
        components = kwargs.pop('components')
        if isinstance(components, str):
            components = (components, )
        for comp in components:
            kwargs.update(dict(components=comp))
            method(self, *args, **kwargs)
        return self
    return decorator

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
    @apply_to_each_component
    def to_2d(self, index, components='traces', length_alingment=None):
        """Docstring."""
        component = components
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
    def dump(self, path, fmt, component='traces', **kwargs):
        """Docstring."""
        if fmt in ['sgy', 'segy']:
            self._dump_segy(path, component, **kwargs)
        else:
            raise NotImplementedError('Unknown file format.')

    def _dump_segy(self, path, component, split=True):
        """Docstring."""
        if split:
            return self._dump_splitted_segy(path, component)
        else:
            return self._dump_single_segy(path, component)

    @inbatch_parallel(init="indices", target="threads")
    def _dump_splitted_segy(self, index, path, component):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        data = getattr(self, component)[pos]
        if isinstance(self.index, TraceIndex):
            data = np.atleast_2d(data)

        path = os.path.join(path, str(index) + '.sgy')
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = np.arange(len(data[0]))
        spec.tracecount = len(data)
        sort_by = self.meta[pos]['sorting']
        df = self.index._idf.loc[[index]].reset_index(drop=isinstance(self.index, TraceIndex))
        if sort_by is not None:
            df = (df.sort_values(by=sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else
                                 (sort_by, component))
                  .reset_index(drop=True))

        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        df = df[headers]
        df.columns = [getattr(segyio.TraceField, k) for k in df.columns.droplevel(1)]
        with segyio.create(path, spec) as file:
            file.trace = data
            meta = df.to_dict('index')
            for i, x in enumerate(file.header[:]):
                meta[i][segyio.TraceField.TRACE_SEQUENCE_FILE] = i
                x.update(meta[i])

        return self

    def _dump_single_segy(self, path, component):
        """Docstring."""
        trace_index = TraceIndex(self.index)
        data = getattr(self, component)
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = np.arange(len(data[0]))
        spec.tracecount = len(data)
        df = trace_index._idf # pylint: disable=protected-access
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        df = df[headers]
        df.columns = [getattr(segyio.TraceField, k) for k in df.columns.droplevel(1)]
        with segyio.create(path, spec) as file:
            file.trace = data
            meta = df.to_dict('index')
            for i, x in enumerate(file.header[:]):
                meta[i][segyio.TraceField.TRACE_SEQUENCE_FILE] = i
                x.update(meta[i])

        return self

    @action
    def merge_segy_files(self, component, path, samples):
        """Docstring."""
        segy_index = SegyFilesIndex(self.index, name=component)

        df = segy_index._idf.reset_index() # pylint: disable=protected-access
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = samples
        spec.tracecount = len(df)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        df = df[headers]
        df.columns = [getattr(segyio.TraceField, k) for k in df.columns.droplevel(1)]
        with segyio.create(path, spec) as file:
            i = 0
            for index in segy_index.indices:
                batch = (type(self)(segy_index.create_subset([index]))
                         .load(components=component, sort_by='TRACE_SEQUENCE_FILE'))
                data = np.array([t for item in getattr(batch, component) for t in item])
                file.trace[i: i + len(data)] = data
                meta = df.iloc[i: i + len(data)].to_dict('index')
                for j, x in enumerate(file.header[i: i + len(data)]):
                    meta[i + j][segyio.TraceField.TRACE_SEQUENCE_FILE] = i + j
                    x.update(meta[i + j])

                i += len(data)

        return self

    @action
    def load(self, src=None, fmt=None, components=None, **kwargs):
        """Docstring."""
        if isinstance(self.index, DataFrameIndex):
            return self._load_segy(components=components, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component
    def _load_segy(self, components='traces', sort_by='trace_number', **kwargs):
        """Docstring."""
        component = components
        idf = self.index._idf # pylint: disable=protected-access
        idf['_pos'] = np.arange(len(idf))

        segy_index = SegyFilesIndex(self.index, name=component)
        order = np.hstack([segy_index._idf.loc[i, '_pos'].tolist() for # pylint: disable=protected-access
                           i in segy_index.indices])

        batch = type(self)(segy_index)._load_from_segy_files(component=component, **kwargs) # pylint: disable=protected-access
        all_traces = np.array([t for item in batch.traces for t in item] + [None])[:-1]

        res = np.array([None] * len(self))
        if isinstance(self.index, TraceIndex):
            items = order[[self.get_pos(None, "indices", i) for i in self.indices]]
            res = all_traces[items]
            for i in range(len(self)):
                self.meta[i].update(dict(sorting=None))
        else:
            for i in self.indices:
                ipos = self.get_pos(None, "indices", i)
                df = idf.loc[[i]].reset_index()
                items = order[df.sort_values(by=sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else
                                             (sort_by, component))['_pos'].tolist()]
                res[ipos] = all_traces[items]
            self.meta[ipos].update(dict(sorting=sort_by))

        setattr(self, component, res)
        idf.drop('_pos', axis=1, inplace=True)
        self.index._idf.columns = pd.MultiIndex.from_arrays([idf.columns.get_level_values(0),
                                                             idf.columns.get_level_values(1)])

        return self

    @inbatch_parallel(init="indices", target="threads")
    def _load_from_segy_files(self, index, component='traces', tslice=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        path = index
        trace_seq = self.index._idf.loc[index][('TRACE_SEQUENCE_FILE', component)] # pylint: disable=protected-access
        if tslice is None:
            tslice = slice(None)
        with segyio.open(path, strict=False) as segyfile:
            traces = np.array([segyfile.trace[i - 1][tslice] for i in
                               np.atleast_1d(trace_seq)] + [None])[:-1]

        self.traces[pos] = traces
        self.meta[pos] = dict(sorting=None)
        return self

    @action
    @inbatch_parallel(init="indices", target="threads")
    @apply_to_each_component
    def sort_traces(self, index, components, sort_by):
        """Docstring."""
        component = components
        if not isinstance(self.index, DataFrameIndex):
            raise TypeError("Sorting is not supported for this Index.")

        pos = self.get_pos(None, "indices", index)
        sorting = self.meta[pos]['sorting']
        if sorting == sort_by:
            return

        df = (self.index._idf.loc[[index]] # pylint: disable=protected-access
              .reset_index(drop=isinstance(self.index, TraceIndex))
              .sort_values(by=sorting if sorting not in FILE_DEPENDEND_COLUMNS else
                           (sorting, component)))
        order = np.argsort(df[sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else
                              (sort_by, component)].tolist())
        getattr(self, component)[pos] = getattr(self, component)[pos][order]
        self.meta[pos]['sorting'] = sort_by

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

    def show_slice(self, index, axis=-1, offset=0, figsize=None, save_to=None,
                   dpi=None, component='traces', **kwargs):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, meta = np.atleast_3d(getattr(self, component)[pos]), self.meta[pos]
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
        plt.ylabel(axes_names[1])
        plt.xlabel(axes_names[0])
        plt.axis('auto')
        plt.xlim([0, traces.shape[ax[0]]])
        plt.ylim([traces.shape[ax[1]], 0])
        if save_to is not None:
            plt.savefig(save_to, dpi=dpi)
        plt.show()
