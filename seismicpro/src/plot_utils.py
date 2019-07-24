"""Utils."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, colors as mcolors
from .utils import measure_gain_amplitude

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

def seismic_plot(arrs, wiggle=False, xlim=None, ylim=None, std=1, # pylint: disable=too-many-branches, too-many-arguments
                 pts=None, s=None, scatter_color=None, names=None, figsize=None,
                 save_to=None, dpi=None, line_color=None, title=None, **kwargs):
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
    scatter_color : color, sequence, or sequence of color, optional
        The marker color.
    names : str or array-like, optional
        Title names to identify subplots.
    figsize : array-like, optional
        Output plot size.
    save_to : str or None, optional
        If not None, save plot to given path.
    dpi : int, optional, default: None
        The resolution argument for matplotlib.pyplot.savefig.
    line_color : color, sequence, or sequence of color, optional, default: None
        The trace color.
    title : str
        Plot title.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Multi-column subplots.

    Raises
    ------
    ValueError
        If ```trace_col``` is sequence and it lenght is not equal to the number of traces.
        If dimensions of given ```arrs``` not in [1, 2].

    """
    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    if isinstance(names, str):
        names = (names,)

    line_color = 'k' if line_color is None else line_color
    fig, ax = plt.subplots(1, len(arrs), figsize=figsize, squeeze=False)
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

                if isinstance(line_color, str):
                    line_color = [line_color] * len(offsets)

                if len(line_color) != len(offsets):
                    raise ValueError("Lenght of line_color must be equal to the number of traces.")

                y = np.arange(*ylim)
                for ix, k in enumerate(offsets):
                    x = k + std * arr[k, slice(*ylim)] / np.std(arr)
                    col = line_color[ix]
                    ax[0, i].plot(x, y, '{}-'.format(col))
                    ax[0, i].fill_betweenx(y, k, x, where=(x > k), color=col)

            else:
                ax[0, i].imshow(arr.T, **kwargs)

        elif arr.ndim == 1:
            ax[0, i].plot(arr, **kwargs)
        else:
            raise ValueError('Invalid ndim to plot data.')

        if names is not None:
            ax[0, i].set_title(names[i])

        if arr.ndim == 2:
            ax[0, i].set_ylim([ylim[1], ylim[0]])
            if (not wiggle) or (pts is not None):
                ax[0, i].set_xlim(xlim)

        if arr.ndim == 1:
            plt.xlim(xlim)

        if pts is not None:
            ax[0, i].scatter(*pts, s=s, c=scatter_color)

        ax[0, i].set_aspect('auto')

    if title is not None:
        fig.suptitle(title)
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

def gain_plot(arrs, window=51, xlim=None, ylim=None, figsize=None, names=None, **kwargs):# pylint: disable=too-many-branches
    r"""Gain's graph plots the ratio of the maximum mean value of
    the amplitude to the mean value of the smoothed amplitude at the moment t.

    First of all for each trace the smoothed version calculated by following formula:
        $$Am = \sqrt{\mathcal{H}(Am)^2 + Am^2}, \ where$$
    Am - Amplitude of trace.
    $\mathcal{H}$ - is a Hilbert transformaion.

    Then the average values of the amplitudes (Am) at each time (t) are calculated.
    After it the resulted value received from the following equation:

        $$ G(t) = - \frac{\max{(Am)}}{Am(t)} $$

    Parameters
    ----------
    sample : array-like
        Seismogram.
    window : int, default 51
        Size of smoothing window of the median filter.
    xlim : tuple or list with size 2
        Bounds for plot's x-axis.
    ylim : tuple or list with size 2
        Bounds for plot's y-axis.
    figsize : array-like, optional
        Output plot size.
    names : str or array-like, optional
        Title names to identify subplots.

    Returns
    -------
    Gain's plot.
    """
    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    _, ax = plt.subplots(1, len(arrs), figsize=figsize)
    ax = ax.reshape(-1) if isinstance(ax, np.ndarray) else [ax]

    for ix, sample in enumerate(arrs):
        result = measure_gain_amplitude(sample, window)
        ax[ix].plot(result, range(len(result)), **kwargs)
        if names is not None:
            ax[ix].set_title(names[ix])
        if xlim is None:
            set_xlim = (max(result)-min(result)*.1, max(result)+min(result)*1.1)
        elif isinstance(xlim[0], (int, float)):
            set_xlim = xlim
        elif len(xlim) != len(arrs):
            raise ValueError('Incorrect format for xbounds.')
        else:
            set_xlim = xlim[ix]

        if ylim is None:
            set_ylim = (len(result)+100, -100)
        elif isinstance(ylim[0], (int, float)):
            set_ylim = ylim
        elif len(ylim) != len(arrs):
            raise ValueError('Incorrect format for ybounds.')
        else:
            set_ylim = ylim[ix]

        ax[ix].set_ylim(set_ylim)
        ax[ix].set_xlim(set_xlim)
        ax[ix].set_xlabel('Maxamp/Amp')
        ax[ix].set_ylabel('Time')
    plt.show()

def statistics_plot(arrs, stats, rate=None, figsize=None, names=None,
                    save_to=None, **kwargs):
    """Show seismograms and various trace statistics, e.g. rms amplitude and rms frequency.

    Parameters
    ----------
    arrs : array-like
        Seismogram or sequence of seismograms.
    stats : str, callable or array-like
        Name of statistics in statistics zoo, custom function to be avaluated or array of stats.
    rate : scalar
        Sampling rate for spectral statistics.
    figsize : array-like, optional
        Output plot size.
    names : str or array-like, optional
        Title names to identify subplots.
    save_to : str or None, optional
        If not None, save plot to given path.
    kwargs : dict
        Named argumets to matplotlib.pyplot.imshow.

    Returns
    -------
    Plots of seismorgams and trace statistics.
    """
    def rms_freq(x, rate):
        "Calculate rms frequency."
        spec = abs(np.fft.rfft(x, axis=1))**2
        spec = spec / spec.sum(axis=1).reshape((-1, 1))
        freqs = np.fft.rfftfreq(len(x[0]), d=rate)
        return  np.sqrt((freqs**2 * spec).sum(axis=1))

    statistics_zoo = dict(ma_ampl=lambda x, *args: np.mean(abs(x), axis=1),
                          rms_ampl=lambda x, *args: np.sqrt(np.mean(x**2, axis=1)),
                          std_ampl=lambda x, *args: np.std(x, axis=1),
                          rms_freq=rms_freq)

    if isinstance(arrs, np.ndarray) and arrs.ndim == 2:
        arrs = (arrs,)

    if isinstance(stats, str) or callable(stats):
        stats = (stats,)

    if isinstance(names, str):
        names = (names,)

    _, ax = plt.subplots(2, len(arrs), figsize=figsize, squeeze=False)
    for i, arr in enumerate(arrs):
        for k in stats:
            if isinstance(k, str):
                func, label = statistics_zoo[k], k
            else:
                func, label = k, k.__name__

            ax[0, i].plot(func(arr, rate), label=label)

        ax[0, i].legend()
        ax[0, i].set_xlim([0, len(arr)])
        ax[0, i].set_aspect('auto')
        ax[0, i].set_title(names[i] if names is not None else '')
        ax[1, i].imshow(arr.T, **kwargs)
        ax[1, i].set_aspect('auto')

    if save_to is not None:
        plt.savefig(save_to)

    plt.show()

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
        color = list(mcolors.CSS4_COLORS.keys())
    df_len = len(df['config'].unique())
    replace = False if len(color) > df_len else True
    chosen_colors = np.random.choice(color, replace=replace, size=df_len)

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

def draw_histogram(df, layout, n_last):
    """Draw histogram of following attribute.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : str
        string where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    n_last : int, optional
        The number of iterations at the end of which the averaging takes place.
    """
    name, attr = layout.split('/')
    max_iter = df['iteration'].max()
    mean_val = df[(df['iteration'] > max_iter - n_last) & (df['name'] == name)].groupby('repetition').mean()[attr]
    plt.figure(figsize=(8, 6))
    plt.title('Histogram of {}'.format(attr))
    plt.hist(mean_val)
    plt.axvline(mean_val.mean(), color='b', linestyle='dashed', linewidth=1, label='mean {}'.format(attr))
    plt.legend()
    plt.show()
    print('Average value (Median) is {:.4}\nStd is {:.4}'.format(mean_val.median(), mean_val.std()))

def show_1d_heatmap(idf, figsize=None, save_to=None, dpi=300, **kwargs):
    """Plot point distribution within 1D bins.

    Parameters
    ----------
    idf : pandas.DataFrame
        Index DataFrame.
    figsize : tuple
        Output figure size.
    save_to : str, optional
        If given, save plot to the path specified.
    dpi : int
        Resolution for saved figure.
    kwargs : dict
        Named argumets for ```matplotlib.pyplot.imshow```.

    Returns
    -------
    Heatmap plot.
    """
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

    heatmap = plt.imshow(hist, **kwargs)
    plt.colorbar(heatmap)
    plt.yticks(np.arange(brange[0]), bindf['line'].drop_duplicates().values, fontsize=8)
    plt.xlabel("Bins index")
    plt.ylabel("Line index")
    plt.axes().set_aspect('auto')
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)

    plt.show()

def show_2d_heatmap(idf, figsize=None, save_to=None, dpi=300, **kwargs):
    """Plot point distribution within 2D bins.

    Parameters
    ----------
    idf : pandas.DataFrame
        Index DataFrame.
    figsize : tuple
        Output figure size.
    save_to : str, optional
        If given, save plot to the path specified.
    dpi : int
        Resolution for saved figure.
    kwargs : dict
        Named argumets for ```matplotlib.pyplot.imshow```.

    Returns
    -------
    Heatmap plot.
    """
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([np.array(i.split('/')).astype(int) for i in bin_counts.index])
    brange = np.max(bins, axis=0)

    hist = np.zeros(brange, dtype=int)
    hist[bins[:, 0] - 1, bins[:, 1] - 1] = bin_counts.values

    if figsize is not None:
        plt.figure(figsize=figsize)

    heatmap = plt.imshow(hist.T, origin='lower', **kwargs)
    plt.colorbar(heatmap)
    plt.xlabel('x-Bins')
    plt.ylabel('y-Bins')
    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)
    plt.show()
