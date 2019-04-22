"""Utils."""
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, colors as mcolors
from sklearn import preprocessing
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
        Named argumets to matplotlib.pyplot.imshow.

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

def spectral_statistics(data, rate):
    """Calculate basic statistics (rms, sts, total variance, mode) of trace
    power spectrum.

    Parameters
    ----------
    data : array-like
        Array of traces.
    rate : scalar
        Sampling rate.

    Returns
    -------
    stats : array
        Arrays of rms, sts, total variance, mode for each trace.
    """
    spec = abs(np.fft.rfft(data, axis=1))**2
    var = np.sum(abs(np.diff(spec, axis=1)), axis=1)
    spec = spec / spec.sum(axis=1).reshape((-1, 1))
    freqs = np.fft.rfftfreq(len(data[0]), d=rate)
    peak = freqs[np.argmax(spec, axis=1)]
    mean = (freqs * spec).sum(axis=1)
    mean2 = (freqs**2 * spec).sum(axis=1)
    std = np.sqrt(mean2 - mean**2)
    return (np.sqrt(mean2), std, var, peak)

def time_statistics(data):
    """Calculate basic statistics (rms, sts, total variance, mode) for traces.

    Parameters
    ----------
    data : array-like
        Array of traces.
    rate : scalar
        Sampling rate.

    Returns
    -------
    stats : array
        Arrays of rms, sts, total variance, mode for each trace.
    """
    peak = np.max(abs(data), axis=1)
    mean = np.mean(abs(data), axis=1)
    std = np.std(data, axis=1)
    mean2 = std**2 + mean**2
    var = np.sum(abs(np.diff(data, axis=1)), axis=1)
    return (np.sqrt(mean2), std, var, peak)

def show_statistics(data, domain, iline, xline, rate=None, tslice=None,
                    figsize=None, **kwargs):
    """Show statistics in 2D plots.

    Parameters
    ----------
    data : array-like
        Array of traces.
    domain : str, 'time' or 'frequency'
        Domain to calculate statistics in.
    rate : scalar
        Sampling rate.
    tslice : slice, default to None
        Slice of time samples to select from data.
    iline : array-like
        Array of inline numbers.
    xline : array-like
        Array of crossline numbers.
    figsize : array-like, optional
        Output plot size.
    kwargs : dict
        Named argumets to matplotlib.pyplot.imshow.

    Returns
    -------
    Plots of statistics distribution.
    """
    if tslice is not None:
        data = data[:, tslice]

    if domain == 'time':
        vals = time_statistics(data)
    elif domain == 'frequency':
        vals = spectral_statistics(data, rate)
    else:
        raise ValueError('Unknown domain.')

    titles = ['RMS', 'STD', 'TOTAL VARIATION', 'MODE']
    enc = preprocessing.LabelEncoder()
    x = enc.fit_transform(iline)
    xc = enc.classes_
    y = enc.fit_transform(xline)
    yc = enc.classes_
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    im = np.zeros((len(xc), len(yc)))
    for i, ax in enumerate(axes.reshape(-1)):
        im[x, y] = vals[i]
        plot = ax.imshow(im.T, **kwargs)
        step = len(xc) // 9
        ax.set_xticks(np.arange(0, len(xc), step))
        ax.set_xticklabels(xc[::step])
        step = len(yc) // 9
        ax.set_yticks(np.arange(0, len(yc), step))
        ax.set_yticklabels(yc[::step])
        ax.set_aspect('auto')
        ax.set_xlabel('INLINE'), ax.set_ylabel('CROSSLINE') # pylint: disable=expression-not-assigned
        ax.set_title(titles[i])
        fig.colorbar(plot, ax=ax)

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
    df[getattr(segyio.TraceField, 'TRACE_SEQUENCE_FILE')] = np.arange(len(df)) + 1

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

        for j, h in enumerate(dst.header):
            h.update({segyio.TraceField.TRACE_SEQUENCE_FILE: j + 1})

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

def show_research(df, layout=None, average_repetitions=False, log_scale=False, rolling_window=None, color=None): # pylint: disable=too-many-branches
    """Show plots given by research dataframe.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : list, optional
        list of strings where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    average_repetitions : bool, optional
        If True, then a separate line will be drawn for each repetition
        else one mean line will be drawn for each repetition.
    log_scale : bool, optional
        If True, values will be logarithmised.
    rolling_window : None or int, optional
        Size of rolling window.
    """
    if layout is None:
        layout = []
        for nlabel, ndf in df.groupby("name"):
            ndf = ndf.drop(['config', 'name', 'iteration', 'repetition'], axis=1).dropna(axis=1)
            for attr in ndf.columns.values:
                layout.append('/'.join([str(nlabel), str(attr)]))
    if isinstance(log_scale, bool):
        log_scale = [log_scale] * len(layout)
    if isinstance(rolling_window, int) or (rolling_window is None):
        rolling_window = [rolling_window] * len(layout)
    rolling_window = [x if x is not None else 1 for x in rolling_window]

    if color is None:
        colors = list(mcolors.CSS4_COLORS.keys())
    df_len = len(df['config'].unique())
    replace = False if len(colors) > df_len else True
    chosen_colors = np.random.choice(colors, replace=replace, size=df_len)

    _, ax = plt.subplots(1, len(layout), figsize=(9 * len(layout), 7))
    if len(layout) == 1:
        ax = (ax, )

    for i, (title, log, roll_w) in enumerate(list(zip(*[layout, log_scale, rolling_window]))):
        name, attr = title.split('/')
        ndf = df[df['name'] == name]
        for (clabel, cdf), curr_color in zip(ndf.groupby("config"), chosen_colors):
            cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
            if average_repetitions:
                idf = cdf.groupby('iteration').mean().drop('repetition', axis=1)
                y_values = idf[attr].rolling(roll_w).mean().values
                if log:
                    y_values = np.log(y_values)
                ax[i].plot(idf.index.values, y_values, label=str(clabel), color=curr_color)
            else:
                for repet, rdf in cdf.groupby('repetition'):
                    rdf = rdf.drop('repetition', axis=1)
                    y_values = rdf[attr].rolling(roll_w).mean().values
                    if log:
                        y_values = np.log(y_values)
                    ax[i].plot(rdf['iteration'].values, y_values,
                               label='/'.join([str(repet), str(clabel)]), color=curr_color)
        ax[i].set_xlabel('iteration')
        ax[i].set_title(title)
        ax[i].legend()
    plt.show()

def print_results(df, layout, average_repetitions=False, sort_by=None, ascending=True, n_last=100):
    """ Show results given by research dataframe.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : str
        string where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    average_repetitions : bool, optional
        If True, then a separate values will be written
        else one mean value will be written.
    sort_by : str or None, optional
        If not None, column's name to sort.
    ascending : bool, None
        Same as in ```pd.sort_value```.
    n_last : int, optional
        The number of iterations at the end of which the averaging takes place.

    Returns
    -------
        : DataFrame
        Research results in DataFrame, where indices is a config parameters and colums is `layout` values
    """
    columns = []
    data = []

    name, attr = layout.split('/')
    ndf = df[df['name'] == name]
    if average_repetitions:
        columns.extend([name + '_mean', name + '_std'])
    else:
        columns.extend([name + '_' + str(i) for i in [*ndf['repetition'].unique(), 'mean', 'std']])
    for _, cdf in ndf.groupby("config"):
        cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
        if average_repetitions:
            idf = cdf.groupby('iteration').mean().drop('repetition', axis=1)
            max_iter = idf.index.max()
            idf = idf[idf.index > max_iter - n_last]
            data.append([idf[attr].mean(), idf[attr].std()])
        else:
            rep = []
            for _, rdf in cdf.groupby('repetition'):
                rdf = rdf.drop('repetition', axis=1)
                max_iter = rdf['iteration'].max()
                rdf = rdf[rdf['iteration'] > max_iter - n_last]
                rep.append(rdf[attr].mean())
            data.append([*rep, np.mean(rep), np.std(rep)])

    res_df = pd.DataFrame(data=data, index=df['config'].unique(), columns=columns)
    if sort_by:
        res_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    return res_df
