"""Seismic batch."""
import os
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import segyio

from ..batchflow import action, inbatch_parallel, Batch

from .seismic_index import SegyFilesIndex, TraceIndex, FILE_DEPENDEND_COLUMNS
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
    This method simply wraps ``apply_along_axis`` method by setting the
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
            method = partialmethod(cls.apply_along_axis, func)
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
    def apply_along_axis(self, index, func, src, dst, *args, slice_axis=0, **kwargs):
        """Apply function along specified axis of batch items.

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
        slice_axis : int
            Axis to iterate data over.
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
        dst_data = np.array([func(x, *args, **kwargs) for x in np.rollaxis(src_data, slice_axis)])
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
    def to_2d(self, index, src, dst, length_alignment=None, pad_value=0):
        """Convert array of 1d arrays to 2d array.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        length_alignment : str, optional
            Defines what to do with arrays of diffetent lengths.
            If 'min', all traces will be cutted by minimal array length.
            If 'max', all traces will be padded to maximal array length.
            If None, try to put array to 2d array as is.

        Returns
        -------
        batch : SeismicBatch
            Batch with items converted to 2d arrays.
        """
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        if data is None or len(data) == 0:
            return

        try:
            data_2d = np.vstack(data)
        except ValueError as err:
            if length_alignment is None:
                raise ValueError(str(err) + '\nTry to set length_alingment to \'max\' or \'min\'')
            elif length_alignment == 'min':
                nsamples = min([len(t) for t in data])
            elif length_alignment == 'max':
                nsamples = max([len(t) for t in data])
            else:
                raise NotImplementedError('Unknown length_alingment')
            shape = (len(data), nsamples)
            data_2d = np.full(shape, pad_value)
            for i, arr in enumerate(data):
                data_2d[i, :len(arr)] = arr[:nsamples]

        getattr(self, dst)[pos] = data_2d

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
        pos = self.get_pos(None, src, index)
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
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
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
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.tracecount = self.index.shape[0]
        with segyio.open(segy_index.indices[0], strict=False) as file:
            spec.samples = file.samples

        with segyio.create(path, spec) as dst:
            i = 0
            for index in segy_index.indices:
                with segyio.open(index, strict=False) as src:
                    dst.trace[i: i + src.tracecount] = src.trace

                i += src.tracecount

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
        if isinstance(self.index, TraceIndex):
            return self._load_segy(src=components, dst=components, **kwargs)

        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    @apply_to_each_component
    def _load_segy(self, src, dst, sort_by=None, tslice=None):
        """Load data from segy files.

        Parameters
        ----------
        src : str, array-like
            Component to load.
        dst : str, array-like
            The batch component to put loaded data in.
        sort_by: str, optional
            Sorting order for traces given by header from segyio.TraceField.
            Default to None.
        tslice: slice, optional
            Load a trace subset given by slice.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        segy_index = SegyFilesIndex(self.index, name=src)
        idf = segy_index._idf # pylint: disable=protected-access
        order = np.hstack([np.where(idf.index == i)[0] for i in segy_index.indices])

        batch = type(self)(segy_index)._load_from_segy_file(src=src, dst=dst, tslice=tslice) # pylint: disable=protected-access
        all_traces = np.array([t for item in getattr(batch, dst) for t in item])[np.argsort(order)]
        self.meta[dst] = dict(samples=batch.meta[dst]['samples'])

        idf = self.index._idf # pylint: disable=protected-access
        if idf.index.name is None:
            self.meta[dst]['sorting'] = None
            items = [self.get_pos(None, "indices", i) for i in idf.index]
            res = np.array(list(all_traces[items]) + [None])[:-1]
        else:
            self.meta[dst]['sorting'] = sort_by
            res = np.array([None] * len(self))
            if sort_by is not None:
                keys = idf[sort_by if sort_by not in FILE_DEPENDEND_COLUMNS else (sort_by, src)].values

            for i in self.indices:
                ipos = self.get_pos(None, "indices", i)
                items = np.where(idf.index == i)[0]
                if sort_by is not None:
                    items = items[np.argsort(keys[items])]

                res[ipos] = all_traces[items]

        setattr(self, dst, res)
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
    def slice_traces(self, index, src, dst, slice_obj):
        """
        Slice traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        slice_obj : slice
            Slice to extract from traces.

        Returns
        -------
        batch : SeismicBatch
            Batch with sliced traces.
        """
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        getattr(self, dst)[pos] = data[:, slice_obj]
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def pad_traces(self, index, src, dst, **kwargs):
        """
        Pad traces with ```numpy.pad```.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        kwargs : dict
            Named arguments to ```numpy.pad```.

        Returns
        -------
        batch : SeismicBatch
            Batch with padded traces.
        """
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        pad_width = kwargs['pad_width']
        if isinstance(pad_width, int):
            pad_width = (pad_width, pad_width)

        kwargs['pad_width'] = [(0, 0)] + [pad_width] + [(0, 0)] * (data.ndim - 2)
        getattr(self, dst)[pos] = np.pad(data, **kwargs)
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
        if not isinstance(self.index, TraceIndex):
            raise TypeError("Sorting is not supported for this Index.")

        pos = self.get_pos(None, src, index)
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
        """Scroll and view batch items. Emaple of use:
        ```
        %matplotlib notebook

        fig, tracker = batch.items_viewer('raw', vmin=-cv, vmax=cv, cmap='gray')
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
        ```

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

    def imshow(self, src, index, figsize=None, save_to=None, dpi=None, **kwargs):
        """Show data on a 2D regular raster.

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
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.imshow(data.T, **kwargs)
        plt.title(index)
        plt.axis('auto')
        if save_to is not None:
            plt.savefig(save_to, dpi=dpi)

        plt.show()
