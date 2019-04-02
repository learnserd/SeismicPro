"""Seismic batch tools."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import segyio

from ..batchflow import FilesIndex

DEFAULT_SEGY_HEADERS = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE']
FILE_DEPENDEND_COLUMNS = ['TRACE_SEQUENCE_FILE', 'file_id']


def get_phi(dfr, dfs):
    """Get median inclination for R and S lines."""
    incl = []
    for _, group in dfs.groupby('sline'):
        x, y = group[['x', 'y']].values.T
        incl.append(line_inclination(x, y))
    for _, group in dfr.groupby('rline'):
        x, y = group[['x', 'y']].values.T
        incl.append(line_inclination(x, y))
    return np.median(np.array(incl) % (np.pi / 2))

def random_bins_shift(pts, bin_size, iters):
    """Monte-Carlo best shift estimation."""
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
    """Iterative best shift estimation."""
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

def rot_2d(arr, phi):
    """Rotate vector."""
    c, s = np.cos(phi), np.sin(phi)
    rotm = np.array([[c, -s], [s, c]])
    return np.dot(rotm, arr.T).T

def make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters):
    """Get bins for 1d seismic."""
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

        pts = rot_2d(pts, -_phi)
        ppx, y = pts[:, 0], np.mean(pts[:, 1])

        if origin is None:
            shift = gradient_bins_shift(ppx, bin_size, iters)
            s = shift + bin_size * ((np.min(ppx) - shift) // bin_size)
            _origin = rot_2d(np.array([[s, y]]), _phi)[0]
        else:
            _origin = origin[rline]
            s = rot_2d(_origin.reshape((-1, 2)), -_phi)[0, 0]

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

def make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters):
    """Get bins for 2d seismic."""
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

    pts = rot_2d(dfm[['CDP_X', 'CDP_Y']].values, -phi)

    if origin is None:
        shift = gradient_bins_shift(pts, bin_size, iters)
        s = shift + bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        origin = rot_2d(s.reshape((1, 2)), phi)[0]
    else:
        s = rot_2d(origin.reshape((1, 2)), -phi)[0]

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

def make_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None, iters=10):
    """Get bins for seismic."""
    if isinstance(bin_size, (list, tuple, np.ndarray)):
        df, meta = make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
    else:
        df, meta = make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
    df.columns = pd.MultiIndex.from_arrays([df.columns, [''] * len(df.columns)])
    return df, meta

def build_sps_df(dfr, dfs, dfx):
    """Index traces according to SPS data."""
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

def make_segy_index(filename, extra_headers=None):
    """Index traces according to SEGY data."""
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
            meta[k] = segyfile.attributes(getattr(segyio.TraceField, k))[:]
        meta['TRACE_SEQUENCE_FILE'] = np.arange(1, segyfile.tracecount + 1)
        meta['file_id'] = np.repeat(filename, segyfile.tracecount)
    df = pd.DataFrame(meta)
    return df

def build_segy_df(extra_headers=None, name=None, **kwargs):
    """Build dataframe."""
    index = FilesIndex(**kwargs)
    df = pd.concat([make_segy_index(index.get_fullpath(i), extra_headers) for
                    i in sorted(index.indices)])
    common_cols = list(set(df.columns) - set(FILE_DEPENDEND_COLUMNS))
    df = df[common_cols + FILE_DEPENDEND_COLUMNS]
    df.columns = pd.MultiIndex.from_arrays([common_cols + FILE_DEPENDEND_COLUMNS,
                                            [''] * len(common_cols) + [name] * len(FILE_DEPENDEND_COLUMNS)])
    return df

def show_1d_heatmap(idf, *args, figsize=None, save_to=None, dpi=300, **kwargs):
    """Plot point distribution within 1D bins."""
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([i.split('/') for i in bin_counts.index])

    bindf = pd.DataFrame(bins, columns=['line', 'pos'])
    bindf['line_code'] = bindf['line'].astype('category').cat.codes + 1
    bindf = bindf.astype({'pos': 'int'})
    bindf['counts'] = bin_counts.values
    bindf = bindf.sort_values(by='line')

    brange = np.max(bindf[['line_code', 'pos']].values, axis=0)
    hist = np.zeros(brange, dtype=int)
    hist[bindf['line_code'].values - 1, bindf['pos'].values - 1] = bindf['counts'].values

    if figsize is not None:
        plt.figure(figsize=figsize)

    heatmap = plt.imshow(hist, *args, **kwargs)
    plt.colorbar(heatmap)
    plt.yticks(np.arange(brange[0]), bindf['line'].drop_duplicates().values, fontsize=8)
    plt.xlabel("Bins index")
    plt.ylabel("Line index")
    plt.axes().set_aspect('auto')
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)
    plt.show()

def show_2d_heatmap(idf, *args, figsize=None, save_to=None, dpi=300, **kwargs):
    """Plot point distribution within 2D bins."""
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([np.array(i.split('/')).astype(int) for i in bin_counts.index])
    brange = np.max(bins, axis=0)

    hist = np.zeros(brange, dtype=int)
    hist[bins[:, 0] - 1, bins[:, 1] - 1] = bin_counts.values

    if figsize is not None:
        plt.figure(figsize=figsize)

    heatmap = plt.imshow(hist.T, origin='lower', *args, **kwargs)
    plt.colorbar(heatmap)
    plt.xlabel('x-Bins')
    plt.ylabel('y-Bins')
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)
    plt.show()
