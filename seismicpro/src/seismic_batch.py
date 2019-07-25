"""Seismic batch.""" # pylint: disable=too-many-lines
import os
from textwrap import dedent
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import pywt
import segyio

from ..batchflow import action, inbatch_parallel, Batch, any_action_failed

from .seismic_index import SegyFilesIndex, FieldIndex

from .utils import FILE_DEPENDEND_COLUMNS, partialmethod, write_segy_file, calculate_corrected_field, massive_block
from .plot_utils import IndexTracker, spectrum_plot, seismic_plot, statistics_plot, gain_plot


PICKS_FILE_HEADERS = ['FieldRecord', 'TraceNumber', 'timeOffset']


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
    def decorator(self, *args, src, dst=None, **kwargs):
        """Returned decorator."""
        if isinstance(src, str):
            src = (src, )
        if dst is None:
            dst = src
        elif isinstance(dst, str):
            dst = (dst, )

        res = []
        for isrc, idst in zip(src, dst):
            res.append(method(self, *args, src=isrc, dst=idst, **kwargs))
        return self if isinstance(res[0], SeismicBatch) else res
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
    index : TraceIndex
        Unique identifiers for sets of seismic traces.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : TraceIndex
        Unique identifiers for sets of seismic traces.
    meta : dict
        Metadata about batch components.
    components : tuple
        Array containing all component's name. Updated only by ``_init_component`` function
        if new component comes from ``dst`` or by ``load`` function.

    Note
    ----
    There are only two ways to add a new components to ``components`` attribute.
    1. Using parameter ``components`` in ``load``.
    2. Using parameter ``dst`` with init function named ``_init_component``.
    """
    def __init__(self, index, *args, preloaded=None, **kwargs):
        super().__init__(index, *args, preloaded=preloaded, **kwargs)
        if preloaded is None:
            self.meta = dict()

    def _init_component(self, *args, dst, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args, kwargs
        dst = (dst, ) if isinstance(dst, str) else dst

        for comp in dst:
            self.meta[comp] = self.meta[comp] if comp in self.meta else dict()

            if self.components is None or comp not in self.components:
                self.add_components(comp, init=self.array_of_nones)

        return self.indices

    def _post_filter_by_mask(self, mask, *args, **kwargs):
        """Component filtration using the union of all the received masks.

        Parameters
        ----------
        mask : list
            List of masks if ``src`` is ``str``
            or list of lists if ``src`` is list.

        Returns
        -------
            : SeismicBatch
            New batch class of filtered components.

        Note
        ----
        All components will be changed with given mask and during the proccess,
        new SeismicBatch instance will be created.
        """
        if any_action_failed(mask):
            all_errors = [error for error in mask if isinstance(error, Exception)]
            print(all_errors)
            raise ValueError(all_errors)
        else:
            _ = args
            src = kwargs.get('src', None)
            src = (src, ) if isinstance(src, str) else src

            mask = np.concatenate((np.array(mask)))
            new_idf = self.index.get_df(index=np.hstack((mask)), reset=False)
            new_index = new_idf.index.unique()

            batch_index = type(self.index).from_index(index=new_index, idf=new_idf,
                                                      index_name=self.index.name)

            batch = type(self)(batch_index)
            batch.add_components(self.components)
            batch.meta = self.meta

            for comp in batch.components:
                setattr(batch, comp, np.array([None] * len(batch.index)))

            for i, index in enumerate(new_index):
                for isrc in batch.components:
                    pos = self.get_pos(None, isrc, index)
                    new_data = getattr(self, isrc)[pos][mask[pos]]
                    getattr(batch, isrc)[i] = new_data
        return batch

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def apply_along_axis(self, index, func, *args, src, dst=None, slice_axis=0, **kwargs):
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
    def apply_transform(self, func, *args, src, dst=None, **kwargs):
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
    def band_pass_filter(self, index, *args, src, dst=None, lowcut=None, highcut=None, fs=1, order=5):
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
        _ = args
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
    def to_2d(self, index, *args, src, dst=None, length_alignment=None, pad_value=0):
        """Convert array of 1d arrays to 2d array.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        length_alignment : str, optional
            Defines what to do with arrays of diffetent lengths.
            If 'min', cut the end by minimal array length.
            If 'max', pad the end to maximal array length.
            If None, try to put array to 2d array as is.

        Returns
        -------
        batch : SeismicBatch
            Batch with items converted to 2d arrays.
        """
        _ = args
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
    def dump(self, src, fmt, path, **kwargs):
        """Export data to file.

        Parameters
        ----------
        src : str
            Batch component to dump data from.
        fmt : str
            Output data format.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if fmt.lower() in ['sgy', 'segy']:
            return self._dump_segy(src, path, **kwargs)
        if fmt == 'picks':
            return self._dump_picking(src, path, **kwargs)
        raise NotImplementedError('Unknown format.')

    @action
    def _dump_segy(self, src, path, split=True):
        """Dump data to segy files.

        Parameters
        ----------
        path : str
            Path for output files.
        src : str
            Batch component to dump data from.
        split : bool
            Whether to dump batch items into separate files.

        Returns
        -------
        batch : SeismicBatch
            Unchanged batch.
        """
        if split:
            return self._dump_split_segy(src, path)

        return self._dump_single_segy(src, path)

    @inbatch_parallel(init="indices", target="threads")
    def _dump_split_segy(self, index, src, path):
        """Dump data to segy files."""
        pos = self.get_pos(None, src, index)
        data = np.atleast_2d(getattr(self, src)[pos])

        path = os.path.join(path, str(index) + '.sgy')

        df = self.index.get_df([index], reset=False)
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df.reset_index(drop=self.index.name is None, inplace=True)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
        df.columns = df.columns.droplevel(1)

        write_segy_file(data, df, self.meta[src]['samples'], path)

        return self

    def _dump_single_segy(self, src, path):
        """Dump data to segy file."""
        data = np.vstack(getattr(self, src))

        df = self.index.get_df(reset=False)
        sort_by = self.meta[src]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df = df.loc[self.indices]
        df.reset_index(drop=self.index.name is None, inplace=True)
        headers = list(set(df.columns.levels[0]) - set(FILE_DEPENDEND_COLUMNS))
        segy_headers = [h for h in headers if hasattr(segyio.TraceField, h)]
        df = df[segy_headers]
        df.columns = df.columns.droplevel(1)

        write_segy_file(data, df, self.meta[src]['samples'], path)

        return self

    @action
    def _dump_picking(self, src, path, traces, to_samples, columns=None):
        """Dump picking to file.

        Parameters
        ----------
        src : str
            Source to get picking from.
        path : str
            Output file path.
        traces : str
            Batch component with corresponding traces.
        to_samples : bool
            Should be picks converted to time samples.
        columns: array_like, optional
            Columns to include in the output file. See PICKS_FILE_HEADERS for default format.

        Returns
        -------
        batch : SeismicBatch
            Batch unchanged.
        """
        data = np.concatenate(getattr(self, src))
        if to_samples:
            data = self.meta[traces]['samples'][data]

        if columns is None:
            columns = PICKS_FILE_HEADERS

        df = self.index.get_df(reset=False)
        sort_by = self.meta[traces]['sorting']
        if sort_by is not None:
            df = df.sort_values(by=sort_by)

        df = df.loc[self.indices]
        df['timeOffset'] = data
        df = df.reset_index(drop=self.index.name is None)[columns]
        df.columns = df.columns.droplevel(1)

        for i in [0, 2, 4]:
            df.insert(i, str(i), "")
        df.to_csv(path, index=False, sep='\t', header=False, encoding='ascii', mode='a')
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
        if fmt.lower() in ['sgy', 'segy']:
            return self._load_segy(src=components, dst=components, **kwargs)
        if fmt == 'picks':
            return self._load_picking(components=components)

        return super().load(src=src, fmt=fmt, components=components, **kwargs)

    def _load_picking(self, components):
        """Load picking from file."""
        idf = self.index.get_df(reset=False)
        res = np.split(idf.FIRST_BREAK_TIME.values,
                       np.cumsum(self.index.tracecounts))[:-1]
        self.add_components(components, init=res)
        return self

    @apply_to_each_component
    def _load_segy(self, src, dst, tslice=None):
        """Load data from segy files.

        Parameters
        ----------
        src : str, array-like
            Component to load.
        dst : str, array-like
            The batch component to put loaded data in.
        tslice: slice, optional
            Load a trace subset given by slice.

        Returns
        -------
        batch : SeismicBatch
            Batch with loaded components.
        """
        segy_index = SegyFilesIndex(self.index, name=src)
        sdf = segy_index.get_df()
        sdf['order'] = np.arange(len(sdf))
        order = self.index.get_df().merge(sdf)['order']

        batch = type(self)(segy_index)._load_from_segy_file(src=src, dst=dst, tslice=tslice) # pylint: disable=protected-access
        all_traces = np.concatenate(getattr(batch, dst))[order]
        self.meta[dst] = batch.meta[dst]

        if self.index.name is None:
            res = np.array(list(np.expand_dims(all_traces, 1)) + [None])[:-1]
        else:
            lens = self.index.tracecounts
            res = np.array(np.split(all_traces, np.cumsum(lens)[:-1]) + [None])[:-1]

        self.add_components(dst, init=res)

        return self

    @inbatch_parallel(init="_init_component", target="threads")
    def _load_from_segy_file(self, index, *args, src, dst, tslice=None):
        """Load from a single segy file."""
        _ = src, args
        pos = self.get_pos(None, "indices", index)
        path = index
        trace_seq = self.index.get_df([index])[('TRACE_SEQUENCE_FILE', src)]
        if tslice is None:
            tslice = slice(None)

        with segyio.open(path, strict=False) as segyfile:
            traces = np.atleast_2d([segyfile.trace[i - 1][tslice] for i in
                                    np.atleast_1d(trace_seq).astype(int)])
            samples = segyfile.samples[tslice]
            interval = segyfile.bin[segyio.BinField.Interval]

        getattr(self, dst)[pos] = traces
        if index == self.indices[0]:
            self.meta[dst]['samples'] = samples
            self.meta[dst]['interval'] = interval
            self.meta[dst]['sorting'] = None

        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def slice_traces(self, index, *args, src, slice_obj, dst=None):
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
        _ = args
        pos = self.get_pos(None, src, index)
        data = getattr(self, src)[pos]
        getattr(self, dst)[pos] = data[:, slice_obj]
        return self

    @action
    @inbatch_parallel(init="_init_component", target="threads")
    @apply_to_each_component
    def pad_traces(self, index, *args, src, dst=None, **kwargs):
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
        _ = args
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
    def sort_traces(self, index, *args, src, sort_by, dst=None):
        """Sort traces.

        Parameters
        ----------
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        sort_by: str
            Sorting key.

        Returns
        -------
        batch : SeismicBatch
            Batch with new trace sorting.
        """
        _ = args
        pos = self.get_pos(None, src, index)
        df = self.index.get_df([index])
        order = np.argsort(df[sort_by].tolist())
        getattr(self, dst)[pos] = getattr(self, src)[pos][order]
        if pos == 0:
            self.meta[dst]['sorting'] = sort_by

        return self

    @action
    @inbatch_parallel(init="indices", post='_post_filter_by_mask', target="threads")
    @apply_to_each_component
    def drop_zero_traces(self, index, src, num_zero, **kwargs):
        """Drop traces with sequence of zeros longer than ```num_zero```.

        Parameters
        ----------
        num_zero : int
            Size of the sequence of zeros.
        src : str, array-like
            The batch components to get the data from.

        Returns
        -------
            : SeismicBatch
            Batch without dropped traces.
        """
        _ = kwargs
        pos = self.get_pos(None, src, index)
        traces = getattr(self, src)[pos]
        mask = list()
        for _, trace in enumerate(traces != 0):
            diff_zeros = np.diff(np.append(np.where(trace)[0], len(trace)))
            mask.append(False if len(diff_zeros) == 0 else np.max(diff_zeros) < num_zero)
        return mask

    @action
    @inbatch_parallel(init='_init_component')
    def field_straightening(self, index, speed, src=None, dst=None, num_mean_tr=4, sample_time=None):
        r""" Straightening up the travel time curve with normal grading. Shift for each
        time value calculated by following way:

        $$\vartriangle t = t(0) \left(\left( 1 + \left( \frac{x}{V(t) t(0)}\right)\right)^{1/2} - 1\right)$$

        New amplitude value for t(0) is the mean value of ```num_mean_tr```'s adjacent
        amplitudes from $t(0) + \vartriangle t$.

        Parameters
        ----------
        speed : array or array of arrays
            Speed law for traces.
        src : str, array-like
            The batch components to get the data from.
        dst : str, array-like
            The batch components to put the result in.
        num_mean_tr : int ,optional default 4
            Number of timestamps to meaning new amplitude value.
        sample_time : int, float, optional
            Difference between real time and samples. Note that ```sample_time``` is measured in milliseconds.

        Returns
        -------
            : SeismicBatch
            Traces straightened on the basis of speed and time values.

        Note
        ----
        1. Works only with sorted traces by offset.
        2. Works properly only with FieldIndex with CDP index.

        Raises
        ------
        ValueError : Raise if traces is not sorted by offset.
        """
        dst = src if dst is None else dst
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]

        offset = np.sort(self.index.get_df(index=index)['offset'])
        speed_conc = np.array(speed[:field.shape[1]])

        if self.meta[src]['sorting'] != 'offset':
            raise ValueError('All traces should be sorted by offset not {}'.format(self.meta[src]['sorting']))
        if 'samples' in self.meta[src].keys():
            sample_time = np.diff(self.meta[src]['samples'][:2])[0]
        elif sample_time is None:
            raise ValueError('Sample time should be specified or by self.meta[src] or by sample_time.')

        if len(speed_conc) != field.shape[1]:
            raise ValueError('Speed must have shape equal to trace lenght, not {} but {}'.format(speed_conc.shape[0],
                                                                                                 field.shape[1]))
        t_zero = (np.arange(1, field.shape[1]+1)*sample_time)/1000
        time_range = np.arange(0, field.shape[1])
        new_field = []
        calc_delta = lambda t_z, spd, ofst: t_z*((1 + (ofst/(spd*t_z+1e-6))**2)**.5 - 1)

        for ix, off in enumerate(offset):
            time_x = calc_delta(t_zero, speed_conc, off)
            shift = np.round((time_x*1000)/sample_time).astype(int)
            down_ix = time_range + shift

            left = -int(num_mean_tr/2) + (~num_mean_tr % 2)
            right = left + num_mean_tr
            mean_traces = np.arange(left, right).reshape(-1, 1)

            ix_to_mean = np.zeros((num_mean_tr, *down_ix.shape)) + [down_ix]*num_mean_tr + mean_traces
            ix_to_mean = np.clip(ix_to_mean, 0, time_range[-1]).astype(int)

            new_field.append(np.mean(field[ix][ix_to_mean], axis=0))

        getattr(self, dst)[pos] = np.array(new_field)
        return self

    @action
    def correct_spherical_divergence(self, src, dst, speed, time=None, fun=None, started_point=None, # pylint: disable=too-many-arguments
                                     v_pow=None, t_pow=None, method='Powell', use_for_all=False,
                                     find_params=True, params_comp=None, bounds=None):
        """Correction of spherical divergence with given parameers or with optimal parameters.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        speed : array
            Wave propagation speed depending on the depth.
        time : array, optimal
            Trace time values. By default self.meta[src]['samples'] is used.
        fun : callable, optional
            Function to minimize.
        started_point : array of 2, optional
            Started values for $v_{pow}$ and $t_{pow}$.
        v_pow : float or int, optional
            Speed's power.
        t_pow : float or int, optional
            Time's power.
        method : str, optional
            Minimization method, see ```scipy.optimize.minimize```. Default Powell
        use_for_all : bool, default False
            If true, optimal parameters for first element will be used for all batch,
            else optimal parameters will find for each field separately.
        find_params : bool, default True
            If true, fields will be compensated with founeded parameres by scipy minimize
            function, else will be used parameters from arguments.
        params_comp : None of str, default None
            If str, parameters will be saved in a component with name ```params_comp```.
        bounds : int, default ((0, 5), (0, 5))
            Optimization bounds.

        Returns
        -------
            : SeismicBatch
            Batch of fields with corrected spherical divergence.

        Note
        ----
        Works properly only with FieldIndex.

        Raises
        ------
        ValueError : If Index is not FieldIndex.
        """
        fields = getattr(self, src)
        bounds = ((0, 5), (0, 5)) if bounds is None else bounds

        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex not {}".format(type(self.index)))

        time = self.meta[src]['samples'] if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]

        if find_params:
            if use_for_all:
                field = fields[0]
                args = (field, time, speed)

                func = minimize(fun, started_point, args=args, method=method, bounds=bounds)
                v_pow, t_pow = func.x

                if params_comp is not None:
                    setattr(self, params_comp, np.array([v_pow, t_pow]))
                self._correct_sph_div(src=src, dst=dst, time=time, speed=speed, v_pow=v_pow, t_pow=t_pow)
            else:
                self._find_and_correct_sd(src=src, dst=dst, fun=fun, started_point=started_point,
                                          arr=(time, speed), method=method, params_comp=params_comp, bounds=bounds)
        else:
            if None in [v_pow, t_pow]:
                raise ValueError("pow_t or pow_v can't be None if find_params is False ")
            self._correct_sph_div(src=src, dst=dst, time=time, speed=speed, v_pow=v_pow, t_pow=t_pow)
        return self

    @inbatch_parallel(init='_init_component')
    def _correct_sph_div(self, index, src, dst, time, speed, v_pow, t_pow):
        """Correct spherical divergence with given parameters. """
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]

        correct_field = calculate_corrected_field(field, time, speed, v_pow=v_pow, t_pow=t_pow)

        getattr(self, dst)[pos] = correct_field
        return self

    @inbatch_parallel(init='_init_component')
    def _find_and_correct_sd(self, index, src, dst, fun, started_point, arr, method, params_comp, bounds):
        """Find optimal parameters and correct spherical divergence. """
        pos = self.get_pos(None, src, index)
        field = getattr(self, src)[pos]

        time, speed = arr
        args = (field, *arr)
        func = minimize(fun, started_point, args=args, method=method, bounds=bounds)
        v_pow, t_pow = func.x
        if params_comp is not None:
            if getattr(self, params_comp) is None:
                raise ValueError('```params_comp``` should be an array but got None')
            getattr(self, params_comp)[pos] = np.array([v_pow, t_pow])
        getattr(self, dst)[pos] = calculate_corrected_field(field, time, speed, v_pow=v_pow, t_pow=t_pow)
        return self

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
            Additional keyword arguments for plt.

        Returns
        -------
        fig, tracker
        """
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, getattr(self, src), self.indices,
                               scroll_step=scroll_step, **kwargs)
        return fig, tracker

    def seismic_plot(self, src, index, wiggle=False, xlim=None, ylim=None, std=1, # pylint: disable=too-many-branches, too-many-arguments
                     src_picking=None, s=None, scatter_color=None, figsize=None,
                     save_to=None, dpi=None, line_color=None, title=None, **kwargs):
        """Plot seismic traces.

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with data to show.
        index : same type as batch.indices
            Data index to show.
        wiggle : bool, default to False
            Show traces in a wiggle form.
        xlim : tuple, optionalgit
            Range in x-axis to show.
        ylim : tuple, optional
            Range in y-axis to show.
        std : scalar, optional
            Amplitude scale for traces in wiggle form.
        src_picking : str
            Component with picking data.
        s : scalar or array_like, shape (n, ), optional
            The marker size in points**2.
        scatter_color : color, sequence, or sequence of color, optional
            The marker color.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        dpi : int, optional, default: None
            The resolution argument for matplotlib.pyplot.savefig.
        line_color : color, sequence, or sequence of color, optional, default: None
            The trace color.
        title : str
            Plot title.
        kwargs : dict
            Additional keyword arguments for plot.

        Returns
        -------
        Multi-column subplots.
        """
        pos = self.get_pos(None, 'indices', index)
        if len(np.atleast_1d(src)) == 1:
            src = (src,)

        if src_picking is not None:
            rate = self.meta[src[0]]['interval'] / 1e3
            picking = getattr(self, src_picking)[pos] / rate
            pts_picking = (range(len(picking)), picking)
        else:
            pts_picking = None

        arrs = [getattr(self, isrc)[pos] for isrc in src]
        names = [' '.join([i, str(index)]) for i in src]
        seismic_plot(arrs=arrs, wiggle=wiggle, xlim=xlim, ylim=ylim, std=std,
                     pts=pts_picking, s=s, scatter_color=scatter_color,
                     figsize=figsize, names=names, save_to=save_to,
                     dpi=dpi, line_color=line_color, title=title, **kwargs)
        return self

    def spectrum_plot(self, src, index, frame, max_freq=None,
                      figsize=None, save_to=None, **kwargs):
        """Plot seismogram(s) and power spectrum of given region in the seismogram(s).

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with data to show.
        index : same type as batch.indices
            Data index to show.
        frame : tuple
            List of slices that frame region of interest.
        max_freq : scalar
            Upper frequence limit.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        kwargs : dict
            Named argumets to matplotlib.pyplot.imshow.

        Returns
        -------
        Plot of seismogram(s) and power spectrum(s).
        """
        pos = self.get_pos(None, 'indices', index)
        if len(np.atleast_1d(src)) == 1:
            src = (src,)

        arrs = [getattr(self, isrc)[pos] for isrc in src]
        names = [' '.join([i, str(index)]) for i in src]
        rate = self.meta[src[0]]['interval'] / 1e6
        spectrum_plot(arrs=arrs, frame=frame, rate=rate, max_freq=max_freq,
                      names=names, figsize=figsize, save_to=save_to, **kwargs)
        return self

    def gain_plot(self, src, index, window=51, xlim=None, ylim=None,
                  figsize=None, names=None, **kwargs):
        """Gain's graph plots the ratio of the maximum mean value of
        the amplitude to the mean value of the amplitude at the moment t.

        Parameters
        ----------
        window : int, default 51
            Size of smoothing window of the median filter.
        xlim : tuple or list with size 2
            Bounds for plot's x-axis.
        ylim : tuple or list with size 2
            Bounds for plot's y-axis.
        figsize : array-like, optional
            Output plot size.
        names : str or array-like, optional
            Title names to identify subplots.

        Returns
        -------
        Gain's plot.
        """
        _ = kwargs
        pos = self.get_pos(None, 'indices', index)
        src = (src, ) if isinstance(src, str) else src
        sample = [getattr(self, source)[pos] for source in src]
        gain_plot(sample, window, xlim, ylim, figsize, names, **kwargs)
        return self

    def statistics_plot(self, src, index, stats, figsize=None, save_to=None, **kwargs):
        """Plot seismogram(s) and various trace statistics.

        Parameters
        ----------
        src : str or array of str
            The batch component(s) with data to show.
        index : same type as batch.indices
            Data index to show.
        stats : str, callable or array-like
            Name of statistics in statistics zoo, custom function to be avaluated or array of stats.
        figsize : array-like, optional
            Output plot size.
        save_to : str or None, optional
            If not None, save plot to given path.
        kwargs : dict
            Named argumets to matplotlib.pyplot.imshow.

        Returns
        -------
        Plot of seismogram(s) and power spectrum(s).
        """
        pos = self.get_pos(None, 'indices', index)
        if len(np.atleast_1d(src)) == 1:
            src = (src,)

        arrs = [getattr(self, isrc)[pos] for isrc in src]
        names = [' '.join([i, str(index)]) for i in src]
        rate = self.meta[src[0]]['interval'] / 1e6
        statistics_plot(arrs=arrs, stats=stats, rate=rate, names=names, figsize=figsize,
                        save_to=save_to, **kwargs)
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
        data = np.stack(data)
        dst_data = (data - np.mean(data, axis=2, keepdims=True)) / (np.std(data, axis=2, keepdims=True) + 10 ** -6)
        setattr(self, dst, np.array([i for i in dst_data] + [None])[:-1])
        return self

    @action
    def picking_to_mask(self, src, dst, src_traces='raw'):
        """Convert picking time to the mask.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        dst : str
            The batch components to put the result in.
        src_traces : str
            The batch components which contains traces.

        Returns
        -------
        batch : SeismicBatch
            Batch with the mask corresponds to the picking.
        """
        data = np.concatenate(np.vstack(getattr(self, src)))
        samples = self.meta[src_traces]['samples']
        tick = samples[1] - samples[0]
        data = np.around(data / tick).astype('int')
        batch_size = data.shape[0]
        trace_length = self.raw[0].shape[1]
        ind = tuple(np.array(list(zip(range(batch_size), data))).T)
        ind[1][ind[1] < 0] = 0
        mask = np.zeros((batch_size, trace_length))
        mask[ind] = 1
        dst_data = np.cumsum(mask, axis=1)
        setattr(self, dst, np.array([i for i in dst_data] + [None])[:-1])
        return self

    @action
    def mask_to_pick(self, src, dst, labels=True):
        """Convert the mask to picking time. Piciking time corresponds to the
        begininning of the longest block of consecutive ones in the mask.

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

        dst_data = massive_block(data)
        setattr(self, dst, np.array([i for i in dst_data] + [None])[:-1])
        return self

    @action
    def mcm(self, src, dst, eps=3, l=12):
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
        lead_win[:, l:] = lead_win[:, l:] - lead_win[:, :-l]
        er = lead_win / (long_win + eps)
        self.add_components(dst, init=np.array([i for i in er] + [None])[:-1])
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
        er = np.stack(getattr(self, src))
        er = np.gradient(er, axis=1)
        picking = np.argmax(er, axis=1)
        self.add_components(dst, np.array([i for i in picking] + [None])[:-1])
        return self
