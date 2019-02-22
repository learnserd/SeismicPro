"""Seismic batch."""
import os
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import segyio

from batchflow import action, inbatch_parallel, Batch

from .field_index import SegyFilesIndex, TraceIndex, DataFrameIndex, FILE_DEPENDEND_COLUMNS
from .utils import IndexTracker, partialmethod


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
    Compute {description} for each trace.
    This method simply wraps ``apply_to_trace`` method by setting the
    ``func`` argument to ``{full_name}``.

    Parameters
    ----------
    src : str, optional
        Batch component to get the data from.
    dst : str, optional
        Batch component to put the result in.
    args : misc
        Any additional positional arguments to ``{full_name}``.
    kwargs : misc
        Any additional named arguments to ``{full_name}``.

    Returns
    -------
    batch : SeismicBatch
        Transformed batch. Changes ``dst`` component.
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()

def apply_to_each_component(method):
    """Combine list of src items and list dst items into pairs of src and dst items
    and apply the method to each pair.

    Parameters
    ----------
    method : callable
        Method to be decorated.

    Returns
    -------
    decorator : callable
        Decorated method.
    """
    def decorator(self, *args, **kwargs):
        """Returned decorator."""
        src_list = kwargs.pop('src')
        dst_list = kwargs.pop('dst')
        if isinstance(src_list, str):
            src_list = (src_list, )
        if isinstance(dst_list, str):
            dst_list = (dst_list, )
        for src, dst in list(zip(src_list, dst_list)):
            kwargs.update(dict(src=src, dst=dst))
            method(self, *args, **kwargs)
        return self
    return decorator

def add_actions(actions_dict, template_docstring):
    """Add new actions in ``SeismicBatch`` by setting ``func`` argument in
    ``SeismicBatch.apply_to_each_trace`` method to given callables.

    Parameters
    ----------
    actions_dict : dict
        A dictionary, containing new methods' names as keys and a callable,
        its full name and description for each method as values.
    template_docstring : str
        A string, that will be formatted for each new method from
        ``actions_dict`` using ``full_name`` and ``description`` parameters
        and assigned to its ``__doc__`` attribute.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            docstring = template_docstring.format(full_name=full_name, description=description)
            method = partialmethod(cls.apply_to_trace, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)
        return cls
    return decorator


@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)  # pylint: disable=too-many-public-methods,too-many-instance-attributes
class SeismicBatch(Batch):
    """Batch class for seimsic data. Contains seismic traces, metadata and processing methods.

    Parameters
    ----------
    index : DataFrameIndex
        Unique identifiers for sets of seismic traces.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : DataFrameIndex
        Unique identifiers for sets of seismic traces.
    meta : 1-D ndarray
        Array of dicts with metadata about batch items.
    """
    components = ('meta', )
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded=preloaded)
        if preloaded is None:
            self.meta = dict()

    def _init_component(self, *args, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if isinstance(dst, str):
            dst = (dst,)
        for comp in dst:
            if not hasattr(self, comp):
                setattr(self, comp, np.array([None] * len(self.index)))
            if comp not in self.meta.keys():
                self.meta[comp] = dict()
        return self.indices

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def apply_to_trace(self, index, func, src, dst, *args, **kwargs):
        """Same as np.apply_to_trace.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept a trace as its first argument.
        src : str, array-like
            Batch component name to get the data from.
        dst : str, array-like
            Batch component name to put the result in.
        item_axis : int, default: 0
            Batch item axis to apply ``func`` along.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : SeismicBatch
            Transformed batch. Changes ``dst`` component.
        """
        i = self.get_pos(None, src, index)
        src_data = getattr(self, src)[i]
        dst_data = np.array([func(x, *args, **kwargs) for x in src_data])
        getattr(self, dst)[i] = dst_data

    @action
    @apply_to_each_component
    def apply_transform(self, func, src, dst, *args, **kwargs):
        """Apply a function to each item in the batch.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept an item of ``src`` as its first argument.
        src : str, array-like
            The source to get the data from.
        dst : str, array-like
            The source to put the result in.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : SeismicBatch
            Transformed batch.
        """
        super().apply_transform(func, *args, src=src, dst=dst, **kwargs)
        dst_data = getattr(self, dst)
        setattr(self, dst, np.array([i for i in dst_data] + [None])[:-1])
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def shift_traces(self, index, src, dst, shift_src):
        """Shift all traces by a number of samples.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        lowcut : real, optional
            Lowcut frequency.
        highcut : real, optional
            Highcut frequency.
        order : int
            The order of the filter.
        fs : real
            Sampling rate.

        Returns
        -------
        batch : SeismicBatch
            Batch with filtered traces.
        """
        i = self.get_pos(None, src, index)
        traces = getattr(self, src)[i]
        if isinstance(shift_src, str):
            shifts = getattr(self, shift_src)[i]

        dst_data = np.array([traces[k][max(0, shifts[k]):] for k in range(len(traces))])
        getattr(self, dst)[i] = dst_data

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def band_pass_filter(self, index, src, dst, lowcut=None, highcut=None, fs=1, order=5):
        """Apply a band pass filter.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        lowcut : real, optional
            Lowcut frequency.
        highcut : real, optional
            Highcut frequency.
        order : int
            The order of the filter.
        fs : real
            Sampling rate.

        Returns
        -------
        batch : SeismicBatch
            Batch with filtered traces.
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
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def to_2d(self, index, src, dst, length_alingment=None):
        """Put array of traces to 2d array.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        length_alingment : str, optional
            Defines what to di with traces of diffetent lengths.
            If 'min', all traces will be cutted by minimal trace length.
            If 'max', all traces will be padded to maximal trace length.
            If None, tries to put traces to 2d array as is.

        Returns
        -------
        batch : SeismicBatch
            Batch with arrays of traces converted to 2d arrays.
        """
        pos = self.get_pos(None, "indices", index)
        traces = getattr(self, src)[pos]
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
        getattr(self, dst)[pos] = traces_2d

    @action
    def dump_segy(self, path, src, split=True):
        """Dump data to segy files.

        Parameters
        ----------
        path : str
            Path for output files.
        src : str
            Batch component to dump data from.
        samples : array-like, optional
            Sample times for traces.
        split : bool
            Whether to dump each batch item into a separate file.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if split:
            return self._dump_splitted_segy(path, src)
        return self._dump_single_segy(path, src)

    @inbatch_parallel(init="indices", target="threads")
    def _dump_splitted_segy(self, index, path, src):
        """Dump data to segy files."""
        pos = self.get_pos(None, "indices", index)
        data = getattr(self, src)[pos]
        if isinstance(self.index, TraceIndex):
            data = np.atleast_2d(data)

        path = os.path.join(path, str(index) + '.sgy')
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = self.meta[src]['samples']
        spec.tracecount = len(data)
        sort_by = self.meta[src]['sorting']
        df = self.index._idf.loc[[index]].reset_index(drop=isinstance(self.index, TraceIndex)) # pylint: disable=protected-access
        if sort_by is not None:
            df = (df.sort_values(by=sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else
                                 (sort_by, src))
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

    def _dump_single_segy(self, path, src):
        """Dump data to segy file."""
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            self.sort_traces(src=src, dst=src, sort_by='TRACE_SEQUENCE_FILE')

        trace_index = TraceIndex(self.index)
        data = np.vstack(getattr(self, src))
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = self.meta[src]['samples']
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

        if sort_by is not None:
            self.sort_traces(src=src, dst=src, sort_by=sort_by)

        return self

    @action
    def merge_segy_files(self, component, path):
        """Merge all indexed segy filed into single segy file.

        Parameters
        ----------
        component : str
            Source component for traces.
        path : str
            Path to output file.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        segy_index = SegyFilesIndex(self.index, name=component)
        df = segy_index._idf.reset_index() # pylint: disable=protected-access
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.tracecount = len(df)
        with segyio.open(segy_index.indices[0], strict=False) as file:
            spec.samples = file.samples

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
        """Load data into components.

        Parameters
        ----------
        src : misc, optional
            Source to load components from.
        fmt : str, optional
            Source format.
        components : str or array-like, optional
            Components to load.
        **kwargs: dict
            Any kwargs to be passed to load method.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        if isinstance(self.index, DataFrameIndex):
            return self._load_segy(src=components, dst=components, **kwargs)
        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component
    def _load_segy(self, src, dst, sort_by='TraceNumber', tslice=None):
        """Load data from segy files.

        Parameters
        ----------
        src : str, array-like
            Component to load.
        dst : str, array-like
            The batch component to put loaded data in.
        sort_by: str, optional
            Sorting order for traces given by header from segyio.TraceField.
            Default to TraceNumber.
        tslice: slice, optional
            Load a trace subset given by slice.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        idf = self.index._idf # pylint: disable=protected-access
        idf['_pos'] = np.arange(len(idf))

        segy_index = SegyFilesIndex(self.index, name=src)
        order = np.hstack([segy_index._idf.loc[i, '_pos'].tolist() for # pylint: disable=protected-access
                           i in segy_index.indices])

        batch = type(self)(segy_index)._load_from_segy_file(src=src, dst=dst, tslice=tslice) # pylint: disable=protected-access
        all_traces = np.array([t for item in getattr(batch, dst) for t in item])
        self.meta[dst] = dict(samples=batch.meta[dst]['samples'])

        if isinstance(self.index, TraceIndex):
            items = order[[self.get_pos(None, "indices", i) for i in self.indices]]
            res = all_traces[items]
            self.meta[dst]['sorting'] = None
        else:
            res = np.array([None] * len(self))
            for i in self.indices:
                ipos = self.get_pos(None, "indices", i)
                df = idf.loc[[i]].reset_index()
                items = order[df.sort_values(by=sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else
                                             (sort_by, src))['_pos'].tolist()]
                res[ipos] = all_traces[items]
            self.meta[dst]['sorting'] = sort_by

        setattr(self, dst, res)
        idf.drop('_pos', axis=1, inplace=True)
        self.index._idf.columns = pd.MultiIndex.from_arrays([idf.columns.get_level_values(0), # pylint: disable=protected-access
                                                             idf.columns.get_level_values(1)])

        return self

    @inbatch_parallel(init="_init_component", target="threads")
    def _load_from_segy_file(self, index, src, dst, tslice=None):
        """Load from a single segy file."""
        _ = src
        pos = self.get_pos(None, "indices", index)
        path = index
        trace_seq = self.index._idf.loc[index][('TRACE_SEQUENCE_FILE', src)] # pylint: disable=protected-access
        if tslice is None:
            tslice = slice(None)
        with segyio.open(path, strict=False) as segyfile:
            traces = np.atleast_2d([segyfile.trace[i - 1][tslice] for i in
                                    np.atleast_1d(trace_seq).astype(int)])
            samples = segyfile.samples[tslice]

        getattr(self, dst)[pos] = traces
        if index == self.indices[0]:
            self.meta[dst]['samples'] = samples
            self.meta[dst]['sorting'] = None
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def sort_traces(self, index, src, dst, sort_by):
        """Sort traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        sort_by: str
            Sorting order for traces given by header from segyio.TraceField.
            Default to ''TraceNumber''.

        Returns
        -------
        batch : SeismicBatch
            Batch with sorted traces in components.
        """
        if not isinstance(self.index, DataFrameIndex):
            raise TypeError("Sorting is not supported for this Index.")

        pos = self.get_pos(None, "indices", index)
        sorting = self.meta[src]['sorting']
        if sorting == sort_by:
            return

        df = (self.index._idf.loc[[index]] # pylint: disable=protected-access
              .reset_index(drop=isinstance(self.index, TraceIndex))
              .sort_values(by=sorting if sorting not in FILE_DEPENDEND_COLUMNS else
                           (sorting, src)))
        order = np.argsort(df[sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else
                              (sort_by, src)].tolist())
        getattr(self, dst)[pos] = getattr(self, src)[pos][order]
        self.meta[dst]['sorting'] = sort_by

    def items_viewer(self, src, scroll_step=1, **kwargs):
        """Scroll and view batch items.

        Parameters
        ----------
        src : str
            The batch component with data to show.
        scroll_step : int, default: 1
            Number of batch items scrolled at one time.
        kwargs: dict
            Additional keyword arguments for matplotlib imshow.

        Returns
        -------
        fig, tracker
        """
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, getattr(self, src), self.indices,
                               scroll_step=scroll_step, **kwargs)
        return fig, tracker

    def show_traces(self, src, index, figsize=None, save_to=None, dpi=None, **kwargs):
        """Show traces on a 2D regular raster.

        Parameters
        ----------
        src : str
            The batch component with data to show.
        index : same type as batch.indices
            Data index to show.
        figsize :  tuple of integers, optional, default: None
            Image figsize as in matplotlib.
        save_to : str, default: None
            Path to save image.
        dpi : int, optional, default: None
            The resolution argument for matplotlib savefig.
        kwargs: dict
            Additional keyword arguments for matplotlib imshow.

        Returns
        -------
        """
        pos = self.get_pos(None, "indices", index)
        traces = getattr(self, src)[pos]

        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.imshow(traces.T, **kwargs)
        plt.title(index)
        plt.ylabel('Samples')
        plt.axis('auto')
        if save_to is not None:
            plt.savefig(save_to, dpi=dpi)
        plt.show()
