"""Index for SeismicBatch."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import segyio

from ..batchflow import DatasetIndex, FilesIndex

from .batch_tools import show_1d_heatmap, show_2d_heatmap

DEFAULT_SEGY_HEADERS = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE']
FILE_DEPENDEND_COLUMNS = ['TRACE_SEQUENCE_FILE', 'file_id']


def get_phi(dfr, dfs):
    """Get median inclination for R and S lines."""
    incl = []
    for _, group in dfs.groupby('sline'):
        x, y = group[['x', 'y']].values.T
        if np.std(y) > np.std(x):
            x, y = y, x
        reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        incl.append(reg.coef_[0])
    for _, group in dfr.groupby('rline'):
        x, y = group[['x', 'y']].values.T
        if np.std(y) > np.std(x):
            x, y = y, x
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
        phi = np.radians(phi) # pylint: disable=assignment-from-no-return
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
    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2) / 2.
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


class TraceIndex(DatasetIndex):
    """Base index class."""
    def __init__(self, *args, index_name=None, **kwargs):
        self.meta = {}
        self._idf = None
        self._index_name = index_name
        super().__init__(*args, **kwargs)

    def __str__(self):
        """String representation of the index DataFrame."""
        return print(self._idf)

    @property
    def shape(self):
        """Return a shape of the index DataFrame."""
        return self._idf.shape

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
            if self._index_name is not None:
                idf.set_index(self._index_name, inplace=True)
            self._idf = idf
            return self._idf.index.unique().sort_values()

        df = self.build_df(**kwargs)
        df.reset_index(drop=df.index.name is None, inplace=True)
        if self._index_name is not None:
            df.set_index(self._index_name, inplace=True)

        self._idf = df
        return self._idf.index.unique().sort_values()

    def build_df(self, **kwargs):
        """Build dataframe."""
        if 'dfx' in kwargs.keys():
            return build_sps_df(**kwargs)
        return build_segy_df(**kwargs)

    def merge(self, x, **kwargs):
        """Merge two DataFrameIndex on common columns."""
        idf = self._idf # pylint: disable=protected-access
        xdf = x._idf # pylint: disable=protected-access
        idf.reset_index(drop=idf.index.names[0] is None, inplace=True)
        xdf.reset_index(drop=xdf.index.names[0] is None, inplace=True)
        if np.all(idf.columns.get_level_values(1) == '') or np.all(xdf.columns.get_level_values(1) == ''):
            common = list(set(idf.columns.get_level_values(0).tolist())
                          .intersection(xdf.columns.get_level_values(0)))
            df = idf.merge(xdf, on=common, **kwargs)
        else:
            df = idf.merge(xdf, **kwargs)

        if self._index_name is not None:
            df.set_index(self._index_name, inplace=True)

        return type(self).from_index(index=df.index.unique().sort_values(), idf=df,
                                     index_name=self._index_name)

    def drop_duplicates(self, subset=None, keep='first'):
        """Drop duplicates from DataFrameIndex."""
        df = self._idf.reset_index().drop_duplicates(subset, keep).set_index(self._index_name)
        return type(self).from_index(index=df.index.unique().sort_values(), idf=df,
                                     index_name=self._index_name)

    def build_from_index(self, index, idf):
        """Build index from another index for indices given."""
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """Return a new DataFrameIndex based on the subset of indices given."""
        return type(self).from_index(index=index, idf=self._idf, index_name=self._index_name)


class SegyFilesIndex(TraceIndex):
    """Index segy files."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = ('file_id', kwargs.get('name'))
        super().__init__(*args, **kwargs)


class CustomIndex(TraceIndex):
    """Index any segyio.TraceField attribute."""
    def __init__(self, *args, **kwargs):
        index_name = kwargs['index_name']
        if index_name is not None:
            extra_headers = kwargs['extra_headers'] if 'extra_headers' in kwargs.keys() else []
            kwargs['extra_headers'] = list(set(extra_headers + [index_name]))
        super().__init__(*args, **kwargs)


class KNNIndex(TraceIndex):
    """Index of nearest traces."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'KNN'
        super().__init__(*args, **kwargs)

    def build_df(self, n_neighbors, **kwargs):
        """Build dataframe."""
        extra_headers = kwargs['extra_headers'] if 'extra_headers' in kwargs.keys() else []
        kwargs['extra_headers'] = list(set(extra_headers + ['CDP_X', 'CDP_Y']))
        field_index = FieldIndex(**kwargs)
        dfs = []
        for fid in field_index.indices:
            df = field_index._idf.loc[fid] # pylint: disable=protected-access
            data = np.stack([df['CDP_X'], df['CDP_Y']]).T
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            _, indices = nbrs.fit(data).kneighbors(data)
            if not np.all(indices[:, 0] == np.arange(len(data))):
                raise ValueError("Faild to build KNNIndex. Duplicated CDP.")

            dfs.append(df.iloc[np.hstack(indices)])
        df = pd.concat(dfs).reset_index()
        indices = np.repeat(np.arange(field_index.shape[0]), n_neighbors)
        df['KNN'] = indices
        return df


class FieldIndex(TraceIndex):
    """Index field records."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'FieldRecord'
        super().__init__(*args, **kwargs)


class BinsIndex(TraceIndex):
    """Index bins of CDP."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'bin_id'
        super().__init__(*args, **kwargs)

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
