"""Seismic batch tools."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from sklearn.preprocessing import MinMaxScaler


def show_1d_heatmap(idf, *args, figsize=None, save_to=None, dpi=300, **kwargs):
    """Docstring."""
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
    """Docstring."""
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

@njit(nogil=True)
def nj_sample_crops(traces, pts, size):
    """Docstring."""
    res = np.zeros((len(pts), ) + size, dtype=traces.dtype)
    asize = np.array(size)
    offset = asize // 2
    start = np.zeros(3, dtype=pts.dtype)
    t_stop = np.zeros(3, dtype=pts.dtype)
    c_stop = np.zeros(3, dtype=pts.dtype)
    for i, p in enumerate(pts):
        start[:p.size] = p - offset[:p.size]
        t_stop[:p.size] = p + asize[:p.size] - offset[:p.size]

        t_start = np.maximum(start, 0)
        step = (np.minimum(p + asize[:p.size] - offset[:p.size],
                           np.array(traces.shape)[:p.size]) - t_start[:p.size])

        c_start = np.maximum(-start, 0)
        c_stop[:p.size] = c_start[:p.size] + step

        res[i][c_start[0]: c_stop[0], c_start[1]: c_stop[1], c_start[2]: c_stop[2]] =\
            traces[t_start[0]: t_stop[0], t_start[1]: t_stop[1], t_start[2]: t_stop[2]]

    return res
