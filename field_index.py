import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dataset import DatasetIndex

def get_phi(dfr, dfs):
    """Docstring."""
    incl = []
    for _, group in dfs.groupby('sline'):
        x, y = group['x'].values, group['y'].values
        if np.std(y) > np.std(x):
            reg = LinearRegression().fit(y.reshape((-1, 1)), x)
        else:
            reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        incl.append(reg.coef_[0])
    for _, group in dfr.groupby('rline'):
        x, y = group['x'].values, group['y'].values
        if np.std(y) > np.std(x):
            reg = LinearRegression().fit(y.reshape((-1, 1)), x)
        else:
            reg = LinearRegression().fit(x.reshape((-1, 1)), y)
        incl.append(reg.coef_[0])
    return np.median(np.arctan(incl) % (np.pi / 2)) * 180 / np.pi

def pstd(a, pts, bin_size):
    """Docstring."""
    npts = pts + np.array(a)
    sx = int(np.min(npts[:, 0]) // bin_size)
    sy = int(np.min(npts[:, 1]) // bin_size)
    tx = int(np.max(npts[:, 0]) // bin_size + 1)
    ty = int(np.max(npts[:, 1]) // bin_size + 1)
    h, _, _ = np.histogram2d(npts[:, 0], npts[:, 1],
                             bins=[bin_size * np.arange(sx, tx + 1),
                                   bin_size * np.arange(sy, ty + 1)])
    return np.std(h)

def grid_shift(pts, bin_size, iters=10):
    """Docstring."""
    minv = np.inf
    shift = np.zeros(2)
    for i in range(iters):
        a = bin_size * np.random.random(2)
        v = pstd(a, pts, bin_size)
        if v < minv:
            minv = v
            shift = a
    return shift

def rot_2d(arr, phi):
    """Docstring."""
    c, s = np.cos(phi), np.sin(phi)
    rotm = np.array([[c, -s], [s, c]])
    return np.dot(rotm, arr.T).T

def make_shot_index(dfr, dfs, dfx):
    """Docstring."""
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1
    dfx = pd.DataFrame(dfx.loc[dfx.index.repeat(n_reps)])
    dfx['rid'] = rids
    dfx['channel'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['r2'] = (dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2
    shot_indices = (dfm['sline'].astype(str) + '/' + dfm['sid'].astype(str)).values

    dfm = dfm.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver'], axis=1)

    dfm.index = pd.MultiIndex.from_arrays([shot_indices, np.arange(len(dfm))])

    return dfm

def make_bin_index(dfr, dfs, dfx, bin_size, origin=None, phi=None):
    """Docstring."""
    adjust = False
    rids = np.hstack([np.arange(s, e + 1) for s, e in
                      list(zip(*[dfx['from_receiver'], dfx['to_receiver']]))])
    channels = np.hstack([np.arange(s, e + 1) for s, e in
                          list(zip(*[dfx['from_channel'], dfx['to_channel']]))])
    n_reps = dfx['to_receiver'] - dfx['from_receiver'] + 1
    dfx = pd.DataFrame(dfx.loc[dfx.index.repeat(n_reps)])
    dfx['rid'] = rids
    dfx['channel'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['x_m'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_m'] = (dfm['y_s'] + dfm['y_r']) / 2.

    if origin is None and phi is None:
        phi = get_phi(dfr, dfs)
        origin = np.min(dfm[['x_m', 'y_m']].values, axis=0)
        adjust = True

    vec = rot_2d(dfm[['x_m', 'y_m']].values - origin, -np.radians(phi))
    dfm['x_m2'] = vec[:, 0]
    dfm['y_m2'] = vec[:, 1]

    if adjust:
        shift = grid_shift(dfm[['x_m2', 'y_m2']].values, bin_size)
        dfm[['x_m2', 'y_m2']] = dfm[['x_m2', 'y_m2']].values + shift
        origin -= rot_2d(shift.reshape((1, 2)), -np.radians(phi))[0]


    sx = int(np.min(dfm['x_m2'].values) // bin_size)
    sy = int(np.min(dfm['y_m2'].values) // bin_size)
    tx = int(np.max(dfm['x_m2'].values) // bin_size + 1)
    ty = int(np.max(dfm['y_m2'].values) // bin_size + 1)
    xbins = bin_size * np.arange(sx, tx + 1)
    ybins = bin_size * np.arange(sy, ty + 1)

    dfm['x_index'] = np.digitize(dfm['x_m2'].values, xbins)
    dfm['y_index'] = np.digitize(dfm['y_m2'].values, ybins)

    bin_indices = (dfm['x_index'].astype(str) + '/' + dfm['y_index'].astype(str)).values

    dfm['x_c'] = xbins[dfm['x_index'].values] + xbins[dfm['x_index'].values - 1]
    dfm['y_c'] = ybins[dfm['y_index'].values] + ybins[dfm['y_index'].values - 1]

    dfm['r2'] = (dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2

    dfm = dfm.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver',
                           'x_m2', 'y_m2', 'x_index', 'y_index', 'x_c', 'y_c'], axis=1)

    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))])

    meta = dict(origin=origin, phi=phi)

    return dfm, meta


class FieldIndex(DatasetIndex):
    """Docstring."""
    def __init__(self, *args, **kwargs):
        self._idf = None
        self.meta = None
        super().__init__(*args, **kwargs)

    def build_index(self, index=None, idf=None, dfr=None, dfs=None, dfx=None,
                    bin_size=None, origin=None, phi=None):
        """ Build index. """
        if index is not None:
            return self.build_from_index(index, idf)
        if bin_size is None:
            df = make_shot_index(dfr, dfs, dfx)
        else:
            df, meta = make_bin_index(dfr, dfs, dfx, bin_size, origin, phi)
            self.meta = meta
        self._idf = df
        return df.index.levels[0]

    def build_from_index(self, index, idf):
        """ Build index from another index for indices given. """
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """ Return a new FieldIndex based on the subset of indices given. """
        return type(self).from_index(index=index, idf=self._idf)
