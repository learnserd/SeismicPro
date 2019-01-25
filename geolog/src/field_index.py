"""Docstring."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import segyio

from batchflow import DatasetIndex,FilesIndex

from .batch_tools import show_1d_heatmap, show_2d_heatmap


def get_phi(dfr, dfs):
    """Docstring."""
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
    """Docstring."""
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
    """Docstring."""
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
    """Docstring."""
    c, s = np.cos(phi), np.sin(phi)
    rotm = np.array([[c, -s], [s, c]])
    return np.dot(rotm, arr.T).T


def make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters):
    """Docstring."""
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
    dfm['x_m'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_m'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['az'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    dfm['x_index'] = None
    meta = {}

    for rline, group in dfm.groupby('rline'):
        pts = group[['x_m', 'y_m']].values
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
        meta.update({rline: dict(origin=_origin, phi=np.rad2deg(_phi))})

    bin_indices = (dfm['rline'].astype(str) + '/' + dfm['x_index'].astype(str)).values
    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))],
                                          names=['bin_id', 'trace_id'])

    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2)
    dfm['field_id'] = (dfm['tape'].astype(str) + '/' + dfm['xid'].astype(str)).values

    dfm = dfm.drop(labels=['tape', 'xid', 'from_channel', 'to_channel',
                           'from_receiver', 'to_receiver', 'x_index'], axis=1)

    return dfm, meta

def make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters):
    """Docstring."""
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
    dfx['trace_number'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['x_m'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_m'] = (dfm['y_s'] + dfm['y_r']) / 2.
    dfm['az'] = np.arctan2(dfm['y_r'] - dfm['y_s'], dfm['x_r'] - dfm['x_s'])

    if phi is None:
        phi = get_phi(dfr, dfs)
    else:
        phi = np.radians(phi)
    if phi > 0:
        phi += -np.pi / 2

    pts = rot_2d(dfm[['x_m', 'y_m']].values, -phi)

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

    bin_indices = np.array([ix + '/' + iy for ix, iy in zip(x_index.astype(str), y_index.astype(str))])
    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))],
                                          names=['bin_id', 'trace_id'])

    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2)
    dfm['field_id'] = (dfm['tape'].astype(str) + '/' + dfm['xid'].astype(str)).values

    dfm = dfm.drop(labels=['tape', 'xid', 'from_channel', 'to_channel',
                           'from_receiver', 'to_receiver'], axis=1)

    meta = dict(origin=origin, phi=np.rad2deg(phi))

    return dfm, meta

def make_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters):
    """Docstring."""
    if isinstance(bin_size, (list, tuple, np.ndarray)):
        df, meta = make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
    else:
        df, meta = make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
    return df, meta

def make_sps_field_index(dfr, dfs, dfx, get_file_by_index=None):
    """Docstring."""
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
    dfm['offset'] = np.sqrt((dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2)
    field_id = (dfm['tape'].astype(str) + '/' + dfm['xid'].astype(str)).values

    dfm = dfm.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver'], axis=1)
    dfm['field_id'] = field_id

    dfm.index = pd.MultiIndex.from_arrays([field_id, np.arange(len(dfm))],
                                          names['field_id', 'trace_id'])

    return dfm

def make_segy_field_index(filename):
    """Docstring."""
    with segyio.open(filename, strict=False) as segyfile:
        segyfile.mmap()
        field_record = segyfile.attributes(segyio.TraceField.FieldRecord)[:]
        trace_number = segyfile.attributes(segyio.TraceField.TraceNumber)[:]

    df = pd.DataFrame(dict(trace_number=trace_number))

    df.index = pd.MultiIndex.from_arrays([field_record, np.arange(len(df))],
                                         names=['field_id', 'trace_id'])

    return df

def make_segy_trace_index(filename):
    """Docstring."""
    with segyio.open(filename, strict=False) as segyfile:
        segyfile.mmap()
        field_record = segyfile.attributes(segyio.TraceField.FieldRecord)[:]
        trace_number = segyfile.attributes(segyio.TraceField.TraceNumber)[:]

    df = pd.DataFrame(dict(trace_number=trace_number, field_id=field_record))
    df.index.name = 'trace_id'

    return df


class DataFrameIndex(DatasetIndex):
    """Docstring."""
    def __init__(self, *args, **kwargs):
        self._idf = None
        self.meta = {}
        super().__init__(*args, **kwargs)

    def build_from_index(self, index, idf):
        """ Build index from another index for indices given. """
        if isinstance(idf.index, pd.MultiIndex):
            dfs = [idf.loc[i] for i in index]
            df = pd.concat(dfs)
            df.index = pd.MultiIndex.from_arrays([np.repeat(index, [len(d) for d in dfs])] +
                                                 [df.index.get_level_values(i) for
                                                  i in range(df.index.nlevels)])
            self._idf = df
        else:
            self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """ Return a new FieldIndex based on the subset of indices given. """
        return type(self).from_index(index=index, idf=self._idf)


class SegyFilesIndex(DataFrameIndex):
    """Docstring."""
    def build_index(self, index=None, idf=None, paths_dict=None, **kwargs):
        """ Build index. """
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            else:
                idf = index._idf
                idf = idf.index.to_frame().merge(idf, how='left', left_index=True, right_index=True)
                paths = [paths_dict[field_id] for field_id in idf.field_id]
                idf.index = pd.MultiIndex.from_arrays([paths, idf.field_id, idf.trace_id],
                                                      names=['file_id', 'field_id', 'trace_id'])
                idf.drop(set(['file_id', 'field_id', 'trace_id']).intersection(idf.columns),
                         axis=1, inplace=True)
                self._idf = idf
                self.meta.update(index.meta)
                return idf.index.levels[0].values

        index = FilesIndex(**kwargs)
        dfs = [FieldIndex(segyfile=index.get_fullpath(i))._idf for i in index.indices]
        paths = np.repeat([index.get_fullpath(i) for i in index.indices], [len(df) for df in dfs])
        df = pd.concat(dfs)
        df.trace_id = np.arange(len(df))
        df.index = pd.MultiIndex.from_arrays([paths, df.index.get_level_values('field_id'),
                                              df.index.get_level_values('trace_id')],
                                             names=['file_id', 'field_id', 'trace_id'])
        self._idf = df
        return df.index.levels[0].values


class TraceIndex(DataFrameIndex):
    """Docstring."""
    def build_index(self, index=None, idf=None, segyfile=None):
        """ Build index. """
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            else:
                idf = index._idf
                idf = idf.index.to_frame().merge(idf, how='left', left_index=True, right_index=True)
                idf.index = idf.trace_id
                idf.drop('trace_id', axis=1, inplace=True)
                self._idf = idf
                self.meta.update(index.meta)
                return idf.index.values
        df = make_segy_trace_index(segyfile)
        indices = df.index.values
        self._idf = df
        return np.array(indices)


class FieldIndex(DataFrameIndex):
    """Docstring."""
    def build_index(self, index=None, idf=None, dfr=None, dfs=None, dfx=None, segyfile=None):
        """ Build index. """
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            else:
                idf = index._idf
                idf = idf.index.to_frame().merge(idf, how='left', left_index=True, right_index=True)
                idf.index = pd.MultiIndex.from_arrays([idf.field_id, idf.trace_id],
                                                      names=['field_id', 'trace_id'])
                idf.drop(['trace_id', 'field_id'], axis=1, inplace=True)
                self._idf = idf
                self.meta.update(index.meta)
                return idf.index.levels[0]
        if segyfile is not None:
            df = make_segy_field_index(segyfile)
        else:
            df = make_sps_field_index(dfr, dfs, dfx)
        indices = df.index.levels[0]
        self._idf = df
        return np.array(indices)


class BinsIndex(DataFrameIndex):
    """Docstring."""
    def build_index(self, index=None, idf=None, dfr=None, dfs=None, dfx=None,
                    bin_size=None, origin=None, phi=None, iters=10):
        """ Build index. """
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            idf = index._idf
            idf = idf.index.to_frame().merge(idf, how='left', left_index=True, right_index=True)

            if 'bin_id' not in idf.columns.tolist() + idf.index.names:
                df, meta = make_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
                df = df.index.to_frame().merge(df, how='left', left_index=True, right_index=True)
                df.drop('trace_id', axis=1, inplace=True)
                idf = idf.merge(df,
                                on=list(set(idf.columns.tolist()).intersection(df.columns.tolist())),
                                how='left')
            idf.index = pd.MultiIndex.from_arrays([idf.bin_id, idf.trace_id],
                                                  names=['bin_id', 'trace_id'])
            idf.drop(set(['bin_id', 'trace_id']).intersection(idf.columns),
                     axis=1, inplace=True)
            self._idf = idf
            self.meta.update(dict(bin_size=bin_size))
            return idf.index.levels[0]

        df, meta = make_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
        indices = df.index.levels[0]
        self._idf = df
        self.meta.update(dict(bin_size=bin_size))
        return np.array(indices)

    def show_heatmap(self, figsize=None, save_to=None, dpi=300):
        """Docstring."""
        bin_size = self.meta['bin_size']
        if isinstance(bin_size, (list, tuple, np.ndarray)):
            show_2d_heatmap(self._idf, figsize=figsize, save_to=save_to, dpi=dpi)
        else:
            show_1d_heatmap(self._idf, figsize=figsize, save_to=save_to, dpi=dpi)
