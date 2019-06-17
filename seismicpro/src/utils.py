"""Seismic batch tools."""
import functools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.signal import medfilt, hilbert
import segyio

from . import seismic_index as si
from ..batchflow import FilesIndex

DEFAULT_SEGY_HEADERS = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE']
FILE_DEPENDEND_COLUMNS = ['TRACE_SEQUENCE_FILE', 'file_id']

def partialmethod(func, *frozen_args, **frozen_kwargs):
    """Wrap a method with partial application of given positional and keyword
    arguments.

    Parameters
    ----------
    func : callable
        A method to wrap.
    frozen_args : misc
        Fixed positional arguments.
    frozen_kwargs : misc
        Fixed keyword arguments.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    @functools.wraps(func)
    def method(self, *args, **kwargs):
        """Wrapped method."""
        return func(self, *frozen_args, *args, **frozen_kwargs, **kwargs)
    return method

def spectral_statistics(data, rate, tslice=None):
    """Calculate basic statistics (rms, sts, total variance, mode) of trace
    power spectrum.

    Parameters
    ----------
    data : array-like
        Array of traces.
    rate : scalar
        Sampling rate.

    Returns
    -------
    stats : array
        Arrays of rms, sts, total variance, mode for each trace.
    """
    if tslice is not None:
        data = data[:, tslice]

    spec = abs(np.fft.rfft(data, axis=1))**2
    var = np.sum(abs(np.diff(spec, axis=1)), axis=1)
    spec = spec / spec.sum(axis=1).reshape((-1, 1))
    freqs = np.fft.rfftfreq(len(data[0]), d=rate)
    peak = freqs[np.argmax(spec, axis=1)]
    mean = (freqs * spec).sum(axis=1)
    mean2 = (freqs**2 * spec).sum(axis=1)
    std = np.sqrt(mean2 - mean**2)
    return np.array([np.sqrt(mean2), std, var, peak])

def time_statistics(data, tslice=None):
    """Calculate basic statistics (rms, sts, total variance, mode) for traces.

    Parameters
    ----------
    data : array-like
        Array of traces.
    rate : scalar
        Sampling rate.

    Returns
    -------
    stats : array
        Arrays of rms, sts, total variance, mode for each trace.
    """
    if tslice is not None:
        data = data[:, tslice]

    peak = np.max(abs(data), axis=1)
    mean = np.mean(abs(data), axis=1)
    std = np.std(data, axis=1)
    mean2 = std**2 + mean**2
    var = np.sum(abs(np.diff(data, axis=1)), axis=1)
    res = (np.sqrt(mean2), std, var, peak)
    return np.array([np.sqrt(mean2), std, var, peak])

def write_segy_file(data, df, samples, path, sorting=None, segy_format=1):
    """Write data and headers into SEGY file.

    Parameters
    ----------
    data : array-like
        Array of traces.
    df : DataFrame
        DataFrame with trace headers data.
    samples : array, same length as traces
        Time samples for trace data.
    path : str
        Path to output file.
    sorting : int
        SEGY file sorting.
    format : int
        SEGY file format.

    Returns
    -------
    """
    spec = segyio.spec()
    spec.sorting = sorting
    spec.format = segy_format
    spec.samples = samples
    spec.tracecount = len(data)

    df.columns = [getattr(segyio.TraceField, k) for k in df.columns]
    df[getattr(segyio.TraceField, 'TRACE_SEQUENCE_FILE')] = np.arange(len(df)) + 1

    with segyio.create(path, spec) as file:
        file.trace = data
        meta = df.to_dict('index')
        for i, x in enumerate(file.header[:]):
            x.update(meta[i])

def merge_segy_files(output_path, **kwargs):
    """Merge segy files into a single segy file.

    Parameters
    ----------
    output_path : str
        Path to output file.
    kwargs : dict
        Keyword arguments to index input segy files.

    Returns
    -------
    """
    segy_index = si.SegyFilesIndex(**kwargs, name='data')
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1
    spec.tracecount = sum(segy_index.tracecounts)
    with segyio.open(segy_index.indices[0], strict=False) as file:
        spec.samples = file.samples

    with segyio.create(output_path, spec) as dst:
        i = 0
        for index in segy_index.indices:
            with segyio.open(index, strict=False) as src:
                dst.trace[i: i + src.tracecount] = src.trace
                dst.header[i: i + src.tracecount] = src.header

            i += src.tracecount

        for j, h in enumerate(dst.header):
            h.update({segyio.TraceField.TRACE_SEQUENCE_FILE: j + 1})

def merge_picking_files(output_path, **kwargs):
    """Merge picking files into a single file.

    Parameters
    ----------
    output_path : str
        Path to output file.
    kwargs : dict
        Keyword arguments to index input files.

    Returns
    -------
    """
    files_index = FilesIndex(**kwargs)
    dfs = []
    for i in files_index.indices:
        path = files_index.get_fullpath(i)
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_path, index=False)

def print_results(df, layout, average_repetitions=False, sort_by=None, ascending=True, n_last=100):
    """ Show results given by research dataframe.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : str
        string where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    average_repetitions : bool, optional
        If True, then a separate values will be written
        else one mean value will be written.
    sort_by : str or None, optional
        If not None, column's name to sort.
    ascending : bool, None
        Same as in ```pd.sort_value```.
    n_last : int, optional
        The number of iterations at the end of which the averaging takes place.

    Returns
    -------
        : DataFrame
        Research results in DataFrame, where indices is a config parameters and colums is `layout` values
    """
    columns = []
    data = []
    index = []
    name, attr = layout.split('/')
    ndf = df[df['name'] == name]
    if average_repetitions:
        columns.extend([name + '_mean', name + '_std'])
    else:
        columns.extend([name + '_' + str(i) for i in [*ndf['repetition'].unique(), 'mean', 'std']])
    for config, cdf in ndf.groupby("config"):
        index.append(config)
        cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
        if average_repetitions:
            idf = cdf.groupby('iteration').mean().drop('repetition', axis=1)
            max_iter = idf.index.max()
            idf = idf[idf.index > max_iter - n_last]
            data.append([idf[attr].mean(), idf[attr].std()])
        else:
            rep = []
            for _, rdf in cdf.groupby('repetition'):
                rdf = rdf.drop('repetition', axis=1)
                max_iter = rdf['iteration'].max()
                rdf = rdf[rdf['iteration'] > max_iter - n_last]
                rep.append(rdf[attr].mean())
            data.append([*rep, np.mean(rep), np.std(rep)])

    res_df = pd.DataFrame(data=data, index=index, columns=columns)
    if sort_by:
        res_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    return res_df

def line_inclination(x, y):
    """Get regression line inclination towards x-axis.

    Parameters
    ----------
    x : array-like
        Data x coordinates.
    y : array-like
        Data y coordinates.

    Returns
    -------
    phi : float
        Inclination towards x-axis. The value is within (-pi/2, pi/2) range.
    """
    if np.std(y) < np.std(x):
        reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        return np.arctan(reg.coef_[0])
    reg = LinearRegression().fit(y.reshape((-1, 1)), x)
    if reg.coef_[0] < 0.:
        return -(np.pi / 2) - np.arctan(reg.coef_[0])
    return (np.pi / 2) - np.arctan(reg.coef_[0])

def get_phi(dfr, dfs):
    """Get median absolute inclination for R and S lines.

    Parameters
    ----------
    dfr : pandas.DataFrame
        Data from R file SPS.
    dfs : pandas.DataFrame
        Data from S file SPS.

    Returns
    -------
    phi : float
        Median absolute inclination of R and S lines towards x-axis.
        The value is within (0, pi/2) range.
    """
    incl = []
    for _, group in dfs.groupby('sline'):
        x, y = group[['x', 'y']].values.T
        incl.append(line_inclination(x, y))
    for _, group in dfr.groupby('rline'):
        x, y = group[['x', 'y']].values.T
        incl.append(line_inclination(x, y))
    return np.median(np.array(incl) % (np.pi / 2))

def random_bins_shift(pts, bin_size, iters=100):
    """Monte-Carlo best shift estimation.

    Parameters
    ----------
    pts : array-like
        Point coordinates.
    bin_size : scalar or tuple of scalars
        Bin size of 1D or 2D grid.
    iters : int
        Number of samples.

    Returns
    -------
    shift : float or tuple of floats
        Optimal grid shift from its default origin that is np.min(pts, axis=0).
    """
    t = np.max(pts, axis=0).reshape((-1, 1))
    min_unif = np.inf
    best_shift = np.zeros(pts.ndim)
    for _ in range(iters):
        shift = -bin_size * np.random.random(pts.ndim)
        s = bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        bins = [np.arange(a, b + bin_size, bin_size) for a, b in zip(s + shift, t)]
        if pts.ndim == 2:
            h = np.histogram2d(*pts.T, bins=bins)[0]
        elif pts.ndim == 1:
            h = np.histogram(pts, bins=bins[0])[0]
        else:
            raise ValueError("pts should be ndim = 1 or 2.")

        unif = np.std(h[h > 0])
        if unif < min_unif:
            min_unif = unif
            best_shift = shift

    return best_shift

def gradient_bins_shift(pts, bin_size, max_iters=10, eps=1e-3):
    """Iterative best shift estimation.

    Parameters
    ----------
    pts : array-like
        Point coordinates.
    bin_size : scalar or tuple of scalars
        Bin size of 1D or 2D grid.
    max_iters : int
        Maximal number of iterations.
    eps : float
        Iterations stop criteria.

    Returns
    -------
    shift : float or tuple of floats
        Optimal grid shift from its default origin that is np.min(pts, axis=0).
    """
    t = np.max(pts, axis=0).reshape((-1, 1))
    shift = np.zeros(pts.ndim)
    states = []
    states_std = []
    for _ in range(max_iters):
        s = bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        bins = [np.arange(a, b + bin_size, bin_size) for a, b in zip(s + shift, t)]
        if pts.ndim == 2:
            h = np.histogram2d(*pts.T, bins=bins)[0]
            dif = np.diff(h, axis=0) / 2.
            vmax = np.vstack([np.max(h[i: i + 2], axis=0) for i in range(h.shape[0] - 1)])
            ratio = dif[vmax > 0] / vmax[vmax > 0]
            xshift = bin_size * np.mean(ratio)
            dif = np.diff(h, axis=1) / 2.
            vmax = np.vstack([np.max(h[:, i: i + 2], axis=1) for i in range(h.shape[1] - 1)]).T
            ratio = dif[vmax > 0] / vmax[vmax > 0]
            yshift = bin_size * np.mean(ratio)
            move = np.array([xshift, yshift])
        elif pts.ndim == 1:
            h = np.histogram(pts, bins=bins[0])[0]
            dif = np.diff(h) / 2.
            vmax = np.hstack([np.max(h[i: i + 2]) for i in range(len(h) - 1)])
            ratio = dif[vmax > 0] / vmax[vmax > 0]
            xshift = bin_size * np.mean(ratio)
            move = np.array([xshift])
        else:
            raise ValueError("pts should be ndim = 1 or 2.")

        states.append(shift.copy())
        states_std.append(np.std(h[h > 0]))

        if np.linalg.norm(move) < bin_size * eps:
            break

        shift += move
    if states_std:
        i = np.argmin(states_std)
        return states[i] % bin_size

    return shift

def rotate_2d(arr, phi):
    """Rotate 2D vector counter-clockwise.

    Parameters
    ----------
    arr : array-like
        Vector coordinates.
    phi : radians
        Rotation angle.

    Returns
    -------
    arr : array-like
        Rotated vector.
    """
    c, s = np.cos(phi), np.sin(phi)
    rotm = np.array([[c, -s], [s, c]])
    return np.dot(rotm, arr.T).T

def make_1d_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None,
                      opt='gradient', **kwargs):
    """Get bins for 1d seismic geometry.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : scalar
        Grid bin size.
    origin : dict
        Grid origin for each line.
    phi : dict
        Grid orientation for each line.
    opt : str
        Grid location optimizer.
    kwargs : dict
        Named argumets for optimizer.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with bins indexing.
    """
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1

    dtypes = dfx.dtypes.values
    dfx = pd.DataFrame(dfx.values.repeat(n_reps, axis=0), columns=dfx.columns)
    for i, c in enumerate(dfx.columns):
        dfx[c] = dfx[c].astype(dtypes[i])

    dfx['rid'] = rids
    dfx['trace_number'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['CDP_X'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['CDP_Y'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['azimuth'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    dfm['x_index'] = None
    meta = {}

    for rline, group in dfm.groupby('rline'):
        pts = group[['CDP_X', 'CDP_Y']].values
        if phi is None:
            if np.std(pts[:, 0]) > np.std(pts[:, 1]):
                reg = LinearRegression().fit(pts[:, :1], pts[:, 1])
                _phi = np.arctan(reg.coef_)[0]
            else:
                reg = LinearRegression().fit(pts[:, 1:], pts[:, 0])
                _phi = np.arctan(1. / reg.coef_)[0]
        else:
            _phi = np.radians(phi[rline]) # pylint: disable=assignment-from-no-return

        pts = rotate_2d(pts, -_phi)
        ppx, y = pts[:, 0], np.mean(pts[:, 1])

        if origin is None:
            if opt == 'gradient':
                shift = gradient_bins_shift(ppx, bin_size, **kwargs)
            elif opt == 'monte-carlo':
                shift = random_bins_shift(ppx, bin_size, **kwargs)
            else:
                raise ValueError('Unknown grid optimizer.')

            s = shift + bin_size * ((np.min(ppx) - shift) // bin_size)
            _origin = rotate_2d(np.array([[s, y]]), _phi)[0]
        else:
            _origin = origin[rline]
            s = rotate_2d(_origin.reshape((-1, 2)), -_phi)[0, 0]

        t = np.max(ppx)
        bins = np.arange(s, t + bin_size, bin_size)

        index = np.digitize(ppx, bins)

        dfm.loc[dfm['rline'] == rline, 'x_index'] = index
        meta.update({rline: dict(origin=_origin,
                                 phi=np.rad2deg(_phi),
                                 bin_size=bin_size)})

    dfm['bin_id'] = (dfm['rline'].astype(str) + '/' + dfm['x_index'].astype(str)).values
    dfm.set_index('bin_id', inplace=True)

    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.

    dfm.drop(labels=['from_channel', 'to_channel',
                     'from_receiver', 'to_receiver',
                     'x_index'], axis=1, inplace=True)
    dfm.rename(columns={'x_s': 'SourceX', 'y_s': 'SourceY'}, inplace=True)

    return dfm, meta

def make_2d_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None,
                      opt='gradient', **kwargs):
    """Get bins for 2d seismic geometry.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : tuple
        Grid bin size.
    origin : dict
        Grid origin for each line.
    phi : dict
        Grid orientation for each line.
    opt : str
        Grid location optimizer.
    kwargs : dict
        Named argumets for optimizer.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with bins indexing.
    """
    if bin_size[0] != bin_size[1]:
        raise ValueError('Bins are not square')

    bin_size = bin_size[0]

    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1

    dtypes = dfx.dtypes.values
    dfx = pd.DataFrame(dfx.values.repeat(n_reps, axis=0), columns=dfx.columns)
    for i, c in enumerate(dfx.columns):
        dfx[c] = dfx[c].astype(dtypes[i])

    dfx['rid'] = rids
    dfx['TraceNumber'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['CDP_X'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['CDP_Y'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['azimuth'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    if phi is None:
        phi = get_phi(dfr, dfs)
    else:
        phi = np.radians(phi) # pylint: disable=assignment-from-no-return

    if phi > 0:
        phi += -np.pi / 2

    pts = rotate_2d(dfm[['CDP_X', 'CDP_Y']].values, -phi) # pylint: disable=invalid-unary-operand-type

    if origin is None:
        if opt == 'gradient':
            shift = gradient_bins_shift(pts, bin_size, **kwargs)
        elif opt == 'monte-carlo':
            shift = random_bins_shift(pts, bin_size, **kwargs)
        else:
            raise ValueError('Unknown grid optimizer.')

        s = shift + bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        origin = rotate_2d(s.reshape((1, 2)), phi)[0]
    else:
        s = rotate_2d(origin.reshape((1, 2)), -phi)[0] # pylint: disable=invalid-unary-operand-type

    t = np.max(pts, axis=0)
    xbins, ybins = np.array([np.arange(a, b + bin_size, bin_size) for a, b in zip(s, t)])

    x_index = np.digitize(pts[:, 0], xbins)
    y_index = np.digitize(pts[:, 1], ybins)

    dfm['bin_id'] = np.array([ix + '/' + iy for ix, iy in zip(x_index.astype(str), y_index.astype(str))])
    dfm.set_index('bin_id', inplace=True)

    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.

    dfm = dfm.drop(labels=['from_channel', 'to_channel',
                           'from_receiver', 'to_receiver'], axis=1)
    dfm.rename(columns={'x_s': 'SourceX', 'y_s': 'SourceY'}, inplace=True)
    meta = dict(origin=origin, phi=np.rad2deg(phi), bin_size=(bin_size, bin_size))
    return dfm, meta

def make_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None, **kwargs):
    """Get bins for seismic geometry.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : scalar or tuple of scalars
        Grid bin size.
    origin : dict
        Grid origin for each line.
    phi : dict
        Grid orientation for each line.
    opt : str
        Grid location optimizer.
    kwargs : dict
        Named argumets for optimizer.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with bins indexing.
    """
    if isinstance(bin_size, (list, tuple, np.ndarray)):
        df, meta = make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, **kwargs)
    else:
        df, meta = make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, **kwargs)

    df.columns = pd.MultiIndex.from_arrays([df.columns, [''] * len(df.columns)])
    return df, meta

def build_sps_df(dfr, dfs, dfx):
    """Index traces according to SPS data.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with trace indexing.
    """
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      zip(*[dfx['from_receiver'], dfx['to_receiver']])])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          zip(*[dfx['from_channel'], dfx['to_channel']])])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1

    dfx.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver'],
             axis=1, inplace=True)

    dtypes = dfx.dtypes.values
    dfx = pd.DataFrame(dfx.values.repeat(n_reps, axis=0), columns=dfx.columns)
    for i, c in enumerate(dfx.columns):
        dfx[c] = dfx[c].astype(dtypes[i])

    dfx['rid'] = rids
    dfx['TraceNumber'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['CDP_X'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['CDP_Y'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['azimuth'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])
    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.
    dfm.rename(columns={'x_s': 'SourceX', 'y_s': 'SourceY'}, inplace=True)
    dfm.columns = pd.MultiIndex.from_arrays([dfm.columns, [''] * len(dfm.columns)])

    return dfm

def make_segy_index(filename, extra_headers=None, limits=None):
    """Index traces in a single SEGY file.

    Parameters
    ----------
    filename : str
        Path to SEGY file.
    extra_headers : array-like or str
        Additional headers to put unto DataFrme. If 'all', all headers are included.
    limits : slice or int, default to None
        If int, index only first ```limits``` traces. If slice, index only traces
        within given range. If None, index all traces.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with trace indexing.
    """
    if not isinstance(limits, slice):
        limits = slice(limits)

    with segyio.open(filename, strict=False) as segyfile:
        segyfile.mmap()
        if extra_headers == 'all':
            headers = [h.__str__() for h in segyio.TraceField.enums()]
        elif extra_headers is None:
            headers = DEFAULT_SEGY_HEADERS
        else:
            headers = set(DEFAULT_SEGY_HEADERS + list(extra_headers))

        meta = dict()

        for k in headers:
            meta[k] = segyfile.attributes(getattr(segyio.TraceField, k))[limits]

        meta['file_id'] = np.repeat(filename, segyfile.tracecount)[limits]

    df = pd.DataFrame(meta)
    return df

def build_segy_df(extra_headers=None, name=None, limits=None, **kwargs):
    """Index traces in multiple SEGY files.

    Parameters
    ----------
    extra_headers : array-like or str
        Additional headers to put unto DataFrme. If 'all', all headers are included.
    name : str
        Name that will be associated with indexed traces.
    limits : slice or int, default to None
        If int, index only first ```limits``` traces. If slice, index only traces
        within given range. If None, index all traces.
    kwargs : dict
        Named argumets for ```batchflow.FilesIndex```.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with trace indexing.
    """
    markup_path = kwargs.pop('markup_path', None)
    index = FilesIndex(**kwargs)
    df = pd.concat([make_segy_index(index.get_fullpath(i), extra_headers, limits) for
                    i in sorted(index.indices)])
    if markup_path is not None:
        markup = pd.read_csv(markup_path)
        df = df.merge(markup, how='inner')
    common_cols = list(set(df.columns) - set(FILE_DEPENDEND_COLUMNS))
    df = df[common_cols + FILE_DEPENDEND_COLUMNS]
    df.columns = pd.MultiIndex.from_arrays([common_cols + FILE_DEPENDEND_COLUMNS,
                                            [''] * len(common_cols) + [name] * len(FILE_DEPENDEND_COLUMNS)])
    return df

def calc_v_rms(t, speed):
    r"""Calculate root mean square speed depend on time.
    Value calculated by following formula:

    $$ V_{rms} = \left(\frac{\sum_0^t V^2}{|V|} \right)^{1/2} $$

    Parameters
    ----------
    t : int
        Time value to calculate $V_rms$.

    speed : array
        Speed (V) with time values at each moment.

    Returns
    -------
        : float
        $V_{rms}$
    """
    if t == 0:
        return speed[0]
    return (np.mean(speed[:t]**2))**.5

def calc_sdc(time, speed, v_pow, t_pow):
    """
    Calculate spherical divergence correction (SDC).
    This value has the following formula:
    $$ g(t) = \frac{V_{rms}^{v_{pow}} * t^{t_{pow}}}{V_0} $$

    Here parameters $v_{pow} and t_{pow} is a hyperparameters.
    The quality of the correction depends on them.

    Parameters
    ----------
    time : array
        Trace time values.
    speed : array
        Wave propagation speed depending on the depth.
    v_pow : float or int
        Speed's power.
    t_pow : float or int
        Time's power.

    Returns
    -------
        : float
        Correction value to suppress the spherical divergence.
    """
    correction = (calc_v_rms(time, speed) ** v_pow * time ** t_pow)/speed[0]
    if correction == 0:
        return 1.
    return correction

def time_dep(field, time, speed, v_pow=2, t_pow=1):
    """ Correction of spherical divergence.

    Parameters
    ----------
    field : array or arrays
        Field for correction.
    time : array
        Trace time values.
    speed : array
        Wave propagation speed depending on the depth.
    v_pow : float or int
        Speed's power.
    t_pow : float or int
        Time's power.

    Returns
        : array of arrays
        Corrected field.
    """
    speed = speed[: field.shape[1]]
    new_field = np.zeros_like(field)
    for ix, t in enumerate(time):
        timestamp = field[:, ix]
        new_field[:, ix] = timestamp * calc_sdc(t, speed, v_pow=v_pow,
                                                t_pow=t_pow)
    return new_field

def calculate_sdc_quality(parameters, field, time, speed):
    """Calculate the quality of found parameters.
    The qualiry caluclated as the median of the first order gradient module.

    Parameters
    ----------
    parameters : list of 2
        Power values for speed and time.
    field : array or arrays
        Field for compensation.
    time : array
        Trace time values.
    speed : array
        Wave propagation speed depending on the depth.

    Returns
    -------
        : float
        Error with given parameters.
    """
    v_pow, t_pow = parameters
    new_field = time_dep(field, time=time, speed=speed,
                         v_pow=v_pow, t_pow=t_pow)
    h_sample = []
    for trace in new_field:
        hilb = hilbert(trace).real
        env = (trace**2 + hilb**2)**.5
        h_sample.append(env)

    h_sample = np.array(h_sample)
    mean_sample = np.mean(h_sample, axis=0)
    max_val = np.max(mean_sample)
    dt_val = (-1) * (max_val / mean_sample)
    result = medfilt(dt_val, 51)
    return np.median(np.abs(np.gradient(result)))
