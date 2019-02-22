"""Seismic batch tools."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
