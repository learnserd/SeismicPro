""" Utilities for metrics study and validation"""
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import patches


def get_windowed_spectrogram_dists(smgr, smgl, dist_fn='sum_abs',
                                   time_frame_width=100, noverlap=None, window=('tukey', 0.25)):
    """
    Calculates distances between traces' spectrograms in sliding windows
    Parameters
    ----------
    smgr : np.array of shape (traces count, timestamps)
    smgl : np.array of shape (traces count, timestamps)
        traces to compute spectrograms on

    dist_fn : 'max_abs', 'sum_abs', 'sum_sq' or callable, optional
        function to calculate distance between 2 specrograms for single trace and single time window
        if callable, should accept 2 arrays of shape (traces count, frequencies, segment times)
        and operate on 2-d axis
        Default is 'sum_abs'

    time_frame_width : int, optional
        nperseg for signal.spectrogram
        see ::meth:: scipy.signal.spectrogram

    noverlap : int, optional
    window : str or tuple or array_like, optional
        see ::meth:: scipy.signal.spectrogram

    Returns
    -------
    np.array of shape (traces count, segment times) with distance heatmap

    """
    kwargs = dict(window=window, nperseg=time_frame_width, noverlap=noverlap)
    *_, spgl = signal.spectrogram(smgl, **kwargs)
    *_, spgr = signal.spectrogram(smgr, **kwargs)

    if callable(dist_fn):  # res(sl, sr)
        res = dist_fn(spgl, spgr)
    elif dist_fn == 'max_abs':
        res = np.abs(spgl - spgr).max(axis=1)
    elif dist_fn == 'sum_abs':
        res = np.sum(np.abs(spgl - spgr), axis=1)
    elif dist_fn == 'sum_sq':
        res = np.sum(np.abs(spgl - spgr) ** 2, axis=1)
    else:
        raise NotImplementedError('modes other than max_abs, sum_abs, sum_sq not implemented yet')

    return res


def draw_modifications_dist(modifications, traces_frac=0.1, distances='sum_abs',  # pylint: disable=too-many-arguments
                            vmin=None, vmax=None, figsize=(15, 15),
                            time_frame_width=100, noverlap=None, window=('tukey', 0.25),
                            n_cols=None, fontsize=20, aspect=None):
    """
    Draws seismograms with distances computed relative to 1-st given seismogram

    Parameters
    ----------
    modifications : list of tuples (np.array, str)
        each tuple represents a seismogram and its label
        traces in seismograms should be ordered by absolute offset increasing

    traces_frac : float, optional
        fraction of traces to use to compure metrics

    distances : list of str or callables, or str, or callable, optional
        dist_fn to pass to get_windowed_spectrogram_dists
        if list is given, all corresponding metrics values are computed

    vmin
    vmax
    figsize :
        parameters to pass to pyplot.imshow

    time_frame_width
    noverlap
    window :
        parameters to pass to get_windowed_spectrogram_dists

    n_cols : int or None, optional
        If int, resulting plots are arranged in n_cols collumns, and several rows, if needed
        if None, resulting plots are arranged in one row

    fontsize : int
        fontsize to use in Axes.set_title

    aspect : 'equal', 'auto', or None
        aspect to pass to Axes.set_aspect. If None, set_aspect is not called
    """

    x, y = 1, len(modifications)
    if n_cols is not None:
        x, y = int(np.ceil(y / n_cols)), n_cols

    _, axs = plt.subplots(x, y, figsize=figsize)
    axs = axs.flatten()

    origin, _ = modifications[0]
    n_traces, n_ts = origin.shape
    n_use_traces = int(n_traces*traces_frac)

    if isinstance(distances, str) or callable(distances):
        distances = (distances, )

    for i, (mod, description) in enumerate(modifications):
        distances_strings = []
        for dist_fn in distances:
            dist_m = get_windowed_spectrogram_dists(mod[0:n_use_traces], origin[0:n_use_traces], dist_fn=dist_fn,
                                                    time_frame_width=time_frame_width, noverlap=noverlap, window=window)
            dist = np.mean(dist_m)
            distances_strings.append("{}: {:.4}".format(dist_fn, dist))

        axs[i].imshow(mod.T, vmin=vmin, vmax=vmax, cmap='gray')
        rect = patches.Rectangle((0, 0), n_use_traces, n_ts, edgecolor='r', facecolor='none', lw=1)
        axs[i].add_patch(rect)
        axs[i].set_title("{},\ndistances from original are:\n{}".format(description, '\n'.join(distances_strings)),
                         fontsize=fontsize)
        if aspect:
            axs[i].set_aspect(aspect)


def get_cv(arrs, q=0.95):
    """
    Calculates upper border for data range covered by a colormap in pyplot.imshow
    """
    return np.abs(np.quantile(np.stack(item for item in arrs), q))
