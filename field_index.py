"""Docstring."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dataset import DatasetIndex


def bins_digitize(a, b, p, bin_size):
    """Docstring."""
    phi = np.arctan2(*(b - a)[::-1])
    x = rot_2d((b - a).reshape((-1, 2)), -phi)[0, 0]
    pp = rot_2d((p - a).reshape((-1, 2)), -phi)[:, 0]
    bins = np.arange(0, x + bin_size, bin_size)
    return np.digitize(pp, bins)

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

def grid_shift(pts, bin_size, iters):
    """Docstring."""
    minv = np.inf
    shift = np.zeros(2)
    for _ in range(iters):
        a = bin_size * np.random.random(2)
        v = pstd(a, pts, bin_size)
        if v < minv:
            minv = v
            shift = a
    return -shift

def pstd_1d(a, pts, bin_size):
    """Docstring."""
    npts = pts + a
    sx = int(np.min(npts) // bin_size)
    tx = int(np.max(npts) // bin_size + 1)
    bins = bin_size * np.arange(sx, tx + 1)
    h, _ = np.histogram(npts, bins=bins)
    return np.std(h)

def segment_shift(pts, bin_size, iters):
    """Docstring."""
    minv = np.inf
    shift = 0.
    for _ in range(iters):
        a = bin_size * np.random.random()
        v = pstd_1d(a, pts, bin_size)
        if v < minv:
            minv = v
            shift = a
    return -shift

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

def make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters):
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
    dfm['x_m'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_m'] = (dfm['y_s'] + dfm['y_r']) / 2.

    dfm['x_index'] = None
    meta = {}

    for rline, group in dfm.groupby('rline'):
        pts = group[['x_m', 'y_m']].values
        if phi is None:
            reg = LinearRegression().fit(pts[:, :1], pts[:, 1])
            x_min = pts[np.argmin(pts[:, 0]), 0]
            a = np.array([x_min, reg.predict(x_min)])
            x_max = pts[np.argmax(pts[:, 0]), 0]
            b = np.array([x_max, reg.predict(x_max)])
            _phi = np.arctan2(*(b - a)[::-1])
        else:
            _phi = np.radians(phi[rline])

        pts = rot_2d(pts.reshape((-1, 2)), -_phi)
        px, y = pts[:, 0], np.mean(pts[:, 1])

        if origin is None:
            shift = segment_shift(px, bin_size, iters)
            sx = int((np.min(px) - shift) // bin_size)
            tx = int((np.max(px) - shift) // bin_size + 1)
            bins = shift + bin_size * np.arange(sx, tx + 1)
            _origin = rot_2d(np.array([[bins[0], y]]), _phi)[0]
        else:
            _origin = origin[rline]
            p = rot_2d(_origin.reshape((-1, 2)), -_phi)[0, 0]
            bins = np.arange(p, np.max(px) + bin_size, bin_size)

        index = np.digitize(px, bins)

        dfm.loc[dfm['rline'] == rline, 'x_index'] = index
        meta.update({rline: dict(origin=_origin, phi=np.rad2deg(_phi))})

    bin_indices = (dfm['rline'].astype(str) + '/' + dfm['x_index'].astype(str)).values

    dfm['r2'] = (dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2

    dfm = dfm.drop(labels=['from_channel', 'to_channel',
                           'from_receiver', 'to_receiver', 'x_index'], axis=1)

    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))])

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
    dfx = pd.DataFrame(dfx.loc[dfx.index.repeat(n_reps)])
    dfx['rid'] = rids
    dfx['channel'] = channels
    dfm = (dfx
           .merge(dfs, on=['sline', 'sid'])
           .merge(dfr, on=['rline', 'rid'], suffixes=('_s', '_r')))
    dfm['x_m'] = (dfm['x_s'] + dfm['x_r']) / 2.
    dfm['y_m'] = (dfm['y_s'] + dfm['y_r']) / 2.

    if phi is None:
        phi = get_phi(dfr, dfs)
    else:
        phi = np.radians(phi)
    if phi > 0:
        phi += -np.pi / 2

    if origin is None:
        origin = np.min(dfm[['x_m', 'y_m']].values, axis=0)
        vec = rot_2d(dfm[['x_m', 'y_m']].values, -phi)
        dfm['x_m2'] = vec[:, 0]
        dfm['y_m2'] = vec[:, 1]
        shift = grid_shift(dfm[['x_m2', 'y_m2']].values, bin_size, iters)

        sx = int((dfm['x_m2'].min() - shift[0]) // bin_size)
        sy = int((dfm['y_m2'].min() - shift[1]) // bin_size)
        tx = int((dfm['x_m2'].max() - shift[0]) // bin_size + 1)
        ty = int((dfm['y_m2'].max() - shift[1]) // bin_size + 1)
        xbins = shift[0] + bin_size * np.arange(sx, tx + 1)
        ybins = shift[1] + bin_size * np.arange(sy, ty + 1)

        origin = rot_2d(np.array([[xbins[0], ybins[0]]]), phi)[0]
    else:
        p = rot_2d(origin.reshape((1, 2)), -phi)[0]
        xbins = np.arange(p[0], dfm['x_m2'].max() + bin_size, bin_size)
        ybins = np.arange(p[1], dfm['y_m2'].max() + bin_size, bin_size)

    dfm['x_index'] = np.digitize(dfm['x_m2'].values, xbins)
    dfm['y_index'] = np.digitize(dfm['y_m2'].values, ybins)

    bin_indices = (dfm['x_index'].astype(str) + '/' + dfm['y_index'].astype(str)).values

    dfm['x_c'] = xbins[dfm['x_index'].values] + xbins[dfm['x_index'].values - 1]
    dfm['y_c'] = ybins[dfm['y_index'].values] + ybins[dfm['y_index'].values - 1]

    dfm['r2'] = (dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2

    dfm = dfm.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver',
                           'x_m2', 'y_m2', 'x_index', 'y_index', 'x_c', 'y_c'], axis=1)

    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))])

    meta = dict(origin=origin, phi=np.rad2deg(phi))

    return dfm, meta


class FieldIndex(DatasetIndex):
    """Docstring."""
    def __init__(self, *args, **kwargs):
        self._idf = None
        self.meta = None
        super().__init__(*args, **kwargs)

    def build_index(self, index=None, idf=None, dfr=None, dfs=None, dfx=None,
                    bin_size=None, origin=None, phi=None, iters=10):
        """ Build index. """
        if index is not None:
            return self.build_from_index(index, idf)
        if bin_size is None:
            df = make_shot_index(dfr, dfs, dfx)
        elif isinstance(bin_size, (list, tuple, np.ndarray)):
            df, meta = make_2d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
            self.meta = meta
        else:
            df, meta = make_1d_bin_index(dfr, dfs, dfx, bin_size, origin, phi, iters)
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
