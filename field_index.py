"""Docstring."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dataset import DatasetIndex

from batch_tools import show_1d_heatmap, show_2d_heatmap


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
    sh = []
    hh = []
    for _ in range(max_iters):
        s = bin_size * ((np.min(pts, axis=0) - shift) // bin_size)
        bins = [np.arange(a, b + bin_size, bin_size) for a, b in zip(s + shift, t)]
        if pts.ndim == 2:
            h = np.histogram2d(*pts.T, bins=bins)[0]
            dif = np.diff(h, axis=0) / 2.
            mx = np.vstack([np.max(h[i: i + 2], axis=0) for i in range(h.shape[0] - 1)])
            ratio = dif[mx > 0] / mx[mx > 0]
            xs = bin_size * np.mean(ratio)
            dif = np.diff(h, axis=1) / 2.
            mx = np.vstack([np.max(h[:, i: i + 2], axis=1) for i in range(h.shape[1] - 1)]).T
            ratio = dif[mx > 0] / mx[mx > 0]
            ys = bin_size * np.mean(ratio)
            move = np.array([xs, ys])
        elif pts.ndim == 1:
            h = np.histogram(pts, bins=bins[0])[0]
            dif = np.diff(h) / 2.
            mx = np.hstack([np.max(h[i: i + 2]) for i in range(len(h) - 1)])
            ratio = dif[mx > 0] / mx[mx > 0]
            xs = bin_size * np.mean(ratio)
            move = np.array([xs])
        else:
            raise ValueError("pts should be ndim = 1 or 2.")
        sh.append(shift.copy())
        hh.append(np.std(h))
        if np.linalg.norm(move) < bin_size * eps:
            break
        shift += move
    i = np.argmin(hh)
    return sh[i] % bin_size

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
            if np.std(pts[:, 0]) > np.std(pts[:, 1]):
                reg = LinearRegression().fit(pts[:, :1], pts[:, 1])
                _phi = np.arctan(reg.coef_)[0]
            else:
                reg = LinearRegression().fit(pts[:, 1:], pts[:, 0])
                _phi = np.arctan(1. / reg.coef_)[0]
        else:
            _phi = np.radians(phi[rline])

        pts = rot_2d(pts, -_phi)
        px, y = pts[:, 0], np.mean(pts[:, 1])

        if origin is None:
            shift = gradient_bins_shift(px, bin_size, iters)
            s = shift + bin_size * ((np.min(px) - shift) // bin_size)
            _origin = rot_2d(np.array([[s, y]]), _phi)[0]
        else:
            _origin = origin[rline]
            s = rot_2d(_origin.reshape((-1, 2)), -_phi)[0, 0]

        t = np.max(px)
        bins = np.arange(s, t + bin_size, bin_size)

        index = np.digitize(px, bins)

        dfm.loc[dfm['rline'] == rline, 'x_index'] = index
        meta.update({rline: dict(origin=_origin, phi=np.rad2deg(_phi))})

    bin_indices = (dfm['rline'].astype(str) + '/' + dfm['x_index'].astype(str)).values
    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))])

    dfm['r2'] = (dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2

    dfm = dfm.drop(labels=['from_channel', 'to_channel',
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
    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))])

    dfm['r2'] = (dfm['x_s'] - dfm['x_r'])**2 + (dfm['y_s'] - dfm['y_r'])**2

    dfm = dfm.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver'], axis=1)

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
        if bin_size is not None:
            self.meta.update(dict(bin_size=bin_size))
        return df.index.levels[0]

    def build_from_index(self, index, idf):
        """ Build index from another index for indices given. """
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """ Return a new FieldIndex based on the subset of indices given. """
        return type(self).from_index(index=index, idf=self._idf)

    def show_heatmap(self):
        """Docstring."""
        bin_size = self.meta['bin_size']
        if isinstance(bin_size, (list, tuple, np.ndarray)):
            show_2d_heatmap(self._idf, bin_size[0])
        else:
            show_1d_heatmap(self._idf, bin_size)
