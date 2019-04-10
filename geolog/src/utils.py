"""Utils."""
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import segyio

from . import seismic_index as si
from ..batchflow import FilesIndex


class IndexTracker:
    """Provides onscroll and update methods for matplotlib scroll_event."""
    def __init__(self, ax, frames, frame_names, scroll_step=1, **kwargs):
        self.ax = ax
        self.frames = frames
        self.step = scroll_step
        self.frame_names = frame_names
        self.img_kwargs = kwargs
        self.ind = len(frames) // 2
        self.update()

    def onscroll(self, event):
        """Onscroll method."""
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + self.step, 0, len(self.frames) - 1)
        else:
            self.ind = np.clip(self.ind - self.step, 0, len(self.frames) - 1)
        self.update()

    def update(self):
        """Update method."""
        self.ax.clear()
        img = self.frames[self.ind]
        img = np.squeeze(img)
        if img.ndim == 2:
            self.ax.imshow(img.T, **self.img_kwargs)
        elif img.ndim == 1:
            self.ax.plot(img.T, **self.img_kwargs)
        else:
            raise ValueError('Invalid ndim to plot data.')

        self.ax.set_title('%s' % self.frame_names[self.ind])
        self.ax.set_aspect('auto')
        if img.ndim == 2:
            self.ax.set_ylim([img.shape[1], 0])
            self.ax.set_xlim([0, img.shape[0]])

def partialmethod(func, *frozen_args, **frozen_kwargs):
    """Wrap a method with partial application of given positional and keyword
    arguments.

    Parameters
    ----------
    func : callable
        A method to wrap.
    frozen_args : misc
        Fixed positional arguments.
    frozen_kwargs : misc
        Fixed keyword arguments.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    @functools.wraps(func)
    def method(self, *args, **kwargs):
        """Wrapped method."""
        return func(self, *frozen_args, *args, **frozen_kwargs, **kwargs)
    return method

def seismic_plot(arrs, wiggle=False, xlim=None, ylim=None, std=1, # pylint: disable=too-many-branches, too-many-arguments
                 pts=None, s=None, c=None, names=None, figsize=None,
                 save_to=None, dpi=None, **kwargs):
    """Plot seismic traces.

    Parameters
    ----------
    arrs : array-like
        Arrays of seismic traces to plot.
    wiggle : bool, default to False
        Show traces in a wiggle form.
    xlim : tuple, optional
        Range in x-axis to show.
    ylim : tuple, optional
        Range in y-axis to show.
    std : scalar, optional
        Amplitude scale for traces in wiggle form.
    pts : array_like, shape (n, )
        The points data positions.
    s : scalar or array_like, shape (n, ), optional
        The marker size in points**2.
    c : color, sequence, or sequence of color, optional
        The marker color.
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    save_to : str or None, optional
        If not None, save plot to given path.
    dpi : int, optional, default: None
        The resolution argument for matplotlib.pyplot.savefig.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Multi-column subplots.
    """
    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    if isinstance(names, str):
        names = (names,)

    _, ax = plt.subplots(1, len(arrs), figsize=figsize, squeeze=False)
    for i, arr in enumerate(arrs):
        if not wiggle:
            arr = np.squeeze(arr)

        if xlim is None:
            xlim = (0, len(arr))

        if arr.ndim == 2:
            if ylim is None:
                ylim = (0, len(arr[0]))

            if wiggle:
                offsets = np.arange(*xlim)
                y = np.arange(*ylim)
                for k in offsets:
                    x = k + std * arr[k, slice(*ylim)] / np.std(arr)
                    ax[0, i].plot(x, y, 'k-')
                    ax[0, i].fill_betweenx(y, k, x, where=(x > k), color='k')

            else:
                ax[0, i].imshow(arr.T, **kwargs)

        elif arr.ndim == 1:
            ax[0, i].plot(arr, **kwargs)
        else:
            raise ValueError('Invalid ndim to plot data.')

        if pts is not None:
            ax[0, i].scatter(*pts, s=s, c=c)

        if names is not None:
            ax[0, i].set_title(names[i])

        if arr.ndim == 2:
            plt.ylim([ylim[1], ylim[0]])
            if (not wiggle) or (pts is not None):
                plt.xlim(xlim)

        if arr.ndim == 1:
            plt.xlim(xlim)

        ax[0, i].set_aspect('auto')

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)

    plt.show()

def spectrum_plot(arrs, frame, rate, max_freq=None, names=None,
                  figsize=None, save_to=None, **kwargs):
    """Plot seismogram(s) and power spectrum of given region in the seismogram(s).

    Parameters
    ----------
    arrs : array-like
        Seismogram or sequence of seismograms.
    frame : tuple
        List of slices that frame region of interest.
    rate : scalar
        Sampling rate.
    max_freq : scalar
        Upper frequence limit.
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    save_to : str or None, optional
        If not None, save plot to given path.
    kwargs : dict
        Named argumets to matplotlib.pyplot.imshow

    Returns
    -------
    Plot of seismogram(s) and power spectrum(s).
    """
    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    if isinstance(names, str):
        names = (names,)

    _, ax = plt.subplots(2, len(arrs), figsize=figsize, squeeze=False)
    for i, arr in enumerate(arrs):
        ax[0, i].imshow(arr.T, **kwargs)
        rect = patches.Rectangle((frame[0].start, frame[1].start),
                                 frame[0].stop - frame[0].start,
                                 frame[1].stop - frame[1].start,
                                 edgecolor='r', facecolor='none', lw=2)
        ax[0, i].add_patch(rect)
        ax[0, i].set_title('Seismogram {}'.format(names[i] if names
                                                  is not None else ''))
        ax[0, i].set_aspect('auto')
        spec = abs(np.fft.rfft(arr[frame], axis=1))**2
        freqs = np.fft.rfftfreq(len(arr[frame][0]), d=rate)
        if max_freq is None:
            max_freq = np.inf

        mask = freqs <= max_freq
        ax[1, i].plot(freqs[mask], np.mean(spec, axis=0)[mask], lw=2)
        ax[1, i].set_xlabel('Hz')
        ax[1, i].set_title('Spectrum plot {}'.format(names[i] if names
                                                     is not None else ''))
        ax[1, i].set_aspect('auto')

    if save_to is not None:
        plt.savefig(save_to)

    plt.show()

def write_segy_file(data, df, samples, path, sorting=None, segy_format=1):
    """Write data and headers into SEGY file.

    Parameters
    ----------
    data : array-like
        Array of traces.
    df : DataFrame
        DataFrame with trace headers data.
    samples : array, same length as traces
        Time samples for trace data.
    path : str
        Path to output file.
    sorting : int
        SEGY file sorting.
    format : int
        SEGY file format.

    Returns
    -------
    """
    spec = segyio.spec()
    spec.sorting = sorting
    spec.format = segy_format
    spec.samples = samples
    spec.tracecount = len(data)

    df.columns = [getattr(segyio.TraceField, k) for k in df.columns]

    with segyio.create(path, spec) as file:
        file.trace = data
        meta = df.to_dict('index')
        for i, x in enumerate(file.header[:]):
            x.update(meta[i])

def merge_segy_files(output_path, **kwargs):
    """Merge segy files into a single segy file.

    Parameters
    ----------
    output_path : str
        Path to output file.
    kwargs : dict
        Keyword arguments to index input segy files.

    Returns
    -------
    """
    segy_index = si.SegyFilesIndex(**kwargs, name='data')
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1
    spec.tracecount = sum(segy_index.tracecounts)
    with segyio.open(segy_index.indices[0], strict=False) as file:
        spec.samples = file.samples

    with segyio.create(output_path, spec) as dst:
        i = 0
        for index in segy_index.indices:
            with segyio.open(index, strict=False) as src:
                dst.trace[i: i + src.tracecount] = src.trace
                dst.header[i: i + src.tracecount] = src.header

            i += src.tracecount

def merge_picking_files(output_path, **kwargs):
    """Merge picking files into a single file.

    Parameters
    ----------
    output_path : str
        Path to output file.
    kwargs : dict
        Keyword arguments to index input files.

    Returns
    -------
    """
    files_index = FilesIndex(**kwargs)
    dfs = []
    for i in files_index.indices:
        path = files_index.get_fullpath(i)
        dfs.append(pd.read_csv(path))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(output_path, index=False)
