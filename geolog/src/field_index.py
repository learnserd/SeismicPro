"""Index for SeismicBatch."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import segyio

from batchflow import DatasetIndex, FilesIndex

from .batch_tools import show_1d_heatmap, show_2d_heatmap

DEFAULT_SEGY_HEADERS = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE']
FILE_DEPENDEND_COLUMNS = ['TRACE_SEQUENCE_FILE', 'file_id']


def get_phi(dfr, dfs):
    """Get median inclination for R and S lines."""
    incl = []
    for _, group in dfs.groupby('sline'):
        x, y = group[['x', 'y']].values.T
        if np.std(y) > np.std(x):
            reg = LinearRegression().fit(y.reshape((-1, 1)), x)
        else:
            reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        incl.append(reg.coef_[0])
    for _, group in dfr.groupby('rline'):
        x, y = group[['x', 'y']].values.T
        if np.std(y) > np.std(x):
            reg = LinearRegression().fit(y.reshape((-1, 1)), x)
        else:
            reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        incl.append(reg.coef_[0])
    return np.median(np.arctan(incl) % (np.pi / 2))

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
            unif = np.std(np.histogram2d(*pts.T, bins=bins)[0])
        elif pts.ndim == 1:
            unif = np.std(np.histogram(pts, bins=bins[0])[0])
        else:
            raise ValueError("pts should be ndim = 1 or 2.")
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
        states_std.append(np.std(h))
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
    dfm['x_cdp'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_cdp'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['az'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    dfm['x_index'] = None
    meta = {}

    for rline, group in dfm.groupby('rline'):
        pts = group[['x_cdp', 'y_cdp']].values
        if phi is None:
            if np.std(pts[:, 0]) > np.std(pts[:, 1]):
                reg = LinearRegression().fit(pts[:, :1], pts[:, 1])
                _phi = np.arctan(reg.coef_)[0]
            else:
                reg = LinearRegression().fit(pts[:, 1:], pts[:, 0])
                _phi = np.arctan(1. / reg.coef_)[0]
        else:
            _phi = np.radians(phi[rline])

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
    dfm['x_cdp'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_cdp'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['az'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    if phi is None:
        phi = get_phi(dfr, dfs)
    else:
        phi = np.radians(phi)
    if phi > 0:
        phi += -np.pi / 2

    pts = rot_2d(dfm[['x_cdp', 'y_cdp']].values, -phi)

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

def make_sps_index(dfr, dfs, dfx):
    """Index traces according to SPS data."""
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
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
    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.
    dfm.columns = pd.MultiIndex.from_arrays([dfm.columns, [''] * len(dfm.columns)])

    return dfm

def make_segy_index(filename, extra_headers=None, drop_duplicates=False):
    """Index traces according to SEGY data."""
    with segyio.open(filename, strict=False) as segyfile:
        segyfile.mmap()
        if extra_headers == 'all':
            headers = segyfile.header[0].keys()
        elif extra_headers is None:
            headers = DEFAULT_SEGY_HEADERS
        else:
            headers = list(set(DEFAULT_SEGY_HEADERS + list(extra_headers)))
        meta = dict()
        for k in headers:
            meta[k] = segyfile.attributes(getattr(segyio.TraceField, k))[:]
        meta['TRACE_SEQUENCE_FILE'] = np.arange(1, segyfile.tracecount + 1)
        meta['file_id'] = np.repeat(filename, segyfile.tracecount)
    df = pd.DataFrame(meta)
    if drop_duplicates:
        df.drop_duplicates(subset=['FieldRecord', 'TraceNumber'], keep='last', inplace=True)
    return df


class DataFrameIndex(DatasetIndex):
    """Base index class."""
    def __init__(self, *args, index_name=None, **kwargs):
        self._idf = None
        self.meta = {}
        self.index_name = index_name
        super().__init__(*args, **kwargs)

    def __str__(self):
        """String representation of the index DataFrame."""
        return print(self._idf)

    def head(self, *args, **kwargs):
        """Return the first n rows of the index DataFrame."""
        return self._idf.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Return the last n rows of the index DataFrame."""
        return self._idf.tail(*args, **kwargs)

    def build_index(self, index=None, idf=None, **kwargs):
        """Build index."""
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            idf = index._idf # pylint: disable=protected-access
            idf = idf.reset_index(drop=(idf.index.names[0] is None))
            if self.index_name is not None:
                idf.set_index(self.index_name, inplace=True)
            self._idf = idf
            return self._idf.index.unique().sort_values()#.values

        self._idf = self.build_df(**kwargs)
        return  self._idf.index.unique().sort_values()#.values

    def build_df(self, **kwargs):
        """Build dataframe."""
        raise NotImplementedError("build_df should be defined in child classes")

    def merge(self, x, **kwargs):
        """Merge two DataFrameIndex on common columns."""
        idf = self._idf # pylint: disable=protected-access
        xdf = x._idf # pylint: disable=protected-access
        inames = idf.index.names[0]
        idf.reset_index(drop=idf.index.names[0] is None, inplace=True)
        xdf.reset_index(drop=xdf.index.names[0] is None, inplace=True)
        if np.all(idf.columns.get_level_values(1) == '') or np.all(xdf.columns.get_level_values(1) == ''):
            common = list(set(idf.columns.get_level_values(0).tolist())
                          .intersection(xdf.columns.get_level_values(0)))
            self._idf = idf.merge(xdf, on=common, **kwargs)
        else:
            self._idf = idf.merge(xdf, **kwargs)

        if inames is not None:
            self._idf.set_index(inames, inplace=True)

        return self

    def shuffle(self):
        """Create subset from permuted indices."""
        return self.create_subset(np.random.permutation(self.index))

    def build_from_index(self, index, idf):
        """Build index from another index for indices given."""
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """Return a new FieldIndex based on the subset of indices given."""
        return type(self).from_index(index=index, idf=self._idf)


class SegyFilesIndex(DataFrameIndex):
    """Index segy files."""
    def __init__(self, *args, **kwargs):
        if 'name' in kwargs.keys():
            index_name = ('file_id', kwargs['name'])
        else:
            index_name = ('file_id', None)
        super().__init__(*args, index_name=index_name, **kwargs)

    def build_df(self, extra_headers=None, drop_duplicates=False, name=None, **kwargs):
        """Build dataframe."""
        index = FilesIndex(**kwargs)
        df = pd.concat([make_segy_index(index.get_fullpath(i), extra_headers, drop_duplicates) for
                        i in sorted(index.indices)])
        common_cols = list(set(df.columns) - set(FILE_DEPENDEND_COLUMNS))
        df = df[common_cols + FILE_DEPENDEND_COLUMNS]
        df.columns = pd.MultiIndex.from_arrays([common_cols + FILE_DEPENDEND_COLUMNS,
                                                [''] * len(common_cols) + [name] * len(FILE_DEPENDEND_COLUMNS)])
        df.set_index(('file_id', name), inplace=True)
        self.index_name = ('file_id', name)
        return df


class CustomIndex(DataFrameIndex):
    """Index any segyio.TraceField attribute."""
    def __init__(self, *args, **kwargs):
        index_name = kwargs['index_name']
        if index_name is not None:
            extra_headers = (kwargs['extra_headers'] if 'extra_headers' in kwargs.keys() else [])
            kwargs['extra_headers'] = list(set(extra_headers + [index_name]))
        super().__init__(*args, **kwargs)

    def build_df(self, **kwargs):
        """Build dataframe."""
        return SegyFilesIndex(**kwargs)._idf.set_index(self.index_name) # pylint: disable=protected-access


class TraceIndex(DataFrameIndex):
    """Index traces."""
    def build_df(self, **kwargs):
        """Build dataframe."""
        if 'dfx' in kwargs.keys():
            return make_sps_index(**kwargs)
        return type(self)(SegyFilesIndex(**kwargs))._idf # pylint: disable=protected-access


class FieldIndex(DataFrameIndex):
    """Index field records."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, index_name='FieldRecord', **kwargs)

    def build_df(self, **kwargs):
        """Build dataframe."""
        if 'dfx' in kwargs.keys():
            return make_sps_index(**kwargs).set_index('FieldRecord')
        return type(self)(SegyFilesIndex(**kwargs))._idf # pylint: disable=protected-access


class BinsIndex(DataFrameIndex):
    """Index bins of CDP."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, index_name='bin_id', **kwargs)

    def build_df(self, **kwargs):
        """Build dataframe."""
        df, meta = make_bin_index(**kwargs)
        self.meta.update(meta)
        return df

    def show_heatmap(self, **kwargs):
        """2d histogram of CDP distribution between bins."""
        bin_size = self.meta['bin_size']
        if isinstance(bin_size, (list, tuple, np.ndarray)):
            show_2d_heatmap(self._idf, **kwargs)
        else:
            show_1d_heatmap(self._idf, **kwargs)
