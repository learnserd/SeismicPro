import numpy as np
import pandas as pd

from dataset import DatasetIndex

def make_shot_index(dfr, dfs, dfx):
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

def make_bin_index(dfr, dfs, dfx, origin, phi, bin_size):
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

    phi = -phi * np.pi / 180
    rotm = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    vec = np.dot(rotm, (dfm[['x_m', 'y_m']].values - origin).T).T
    dfm['x_m2'] = vec[:, 0]
    dfm['y_m2'] = vec[:, 1]
    
    if (dfm['x_m2'].min() < 0) or (dfm['y_m2'].min() < 0):
        raise ValueError("Some points are out of gird.")
    
    nx = int(dfm['x_m2'].max() // bin_size) + 1
    ny = int(dfm['y_m2'].max() // bin_size) + 1
    
    xbins = np.linspace(0, bin_size * nx, nx + 1)
    ybins = np.linspace(0, bin_size * ny, ny + 1)
    
    dfm['x_index'] = np.digitize(dfm['x_m2'].values, xbins)
    dfm['y_index'] = np.digitize(dfm['y_m2'].values, ybins)
    
    bin_indices = (dfm['x_index'].astype(str) + '/' + dfm['y_index'].astype(str)).values
    
    dfm['x_c'] = xbins[dfm['x_index'].values] + xbins[dfm['x_index'].values - 1]
    dfm['y_c'] = ybins[dfm['y_index'].values] + ybins[dfm['y_index'].values - 1]
    
    dfm['r2'] = (dfm['x_c'] - dfm['x_m2'])**2 + (dfm['y_c'] - dfm['y_m2'])**2
    
    dfm = dfm.drop(labels=['from_channel', 'to_channel', 'from_receiver', 'to_receiver',
                           'x_m2', 'y_m2', 'x_index', 'y_index', 'x_c', 'y_c'], axis=1)
    
    dfm.index = pd.MultiIndex.from_arrays([bin_indices, np.arange(len(dfm))])
    
    return dfm


class FieldIndex(DatasetIndex):
    def __init__(self, *args, **kwargs):
        self._idf = None
        super().__init__(*args, **kwargs)
    
    def build_index(self, index=None, idf=None, dfr=None, dfs=None, dfx=None,
                    origin=None, phi=None, bin_size=None):
        """ Build index. """
        if index is not None:
            return self.build_from_index(index, idf)
        if origin is None:
            df = make_shot_index(dfr, dfs, dfx)
        else:
            df = make_bin_index(dfr, dfs, dfx, origin, phi, bin_size)
        self._idf = df
        return df.index.levels[0]
    
    def build_from_index(self, index, idf):
        """ Build index from another index for indices given. """
        self._idf = idf.loc[index]
        return index
    
    def create_subset(self, index):
        """ Return a new FieldIndex based on the subset of indices given. """
        return type(self).from_index(index=index, idf=self._idf)
