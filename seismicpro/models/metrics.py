"""Metrics for Seismic procesing tasks."""
import numpy as np
from scipy import signal

from ..batchflow.batchflow.models.metrics import Metrics
from ..src import measure_gain_amplitude

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

def snr(signal, noise, tol=1e-6):
    """Signal-to-noise ratio."""
    nr = np.mean(noise ** 2)
    if nr < tol:
        return 0.
    return np.mean(signal ** 2) / nr

class FieldMetrics(Metrics):
    """Class for seismic field record metrics.
    """
    def __init__(self, targets=None, predictions=None, raw=None, mask=None):
        super().__init__()
        self.targets = targets
        self.predictions = predictions
        self.raw = raw
        self.mask = mask

    def iou(self):
        """Intersection-over-union metric."""
        a = self.targets.astype(float)
        b = self.predictions.astype(float)
        return 2 * np.sum(a * b) / np.sum(a + b)

    def mae(self):
        """Mean absolute error metric."""
        return np.mean(abs(self.targets - self.predictions))

    def mse(self):
        """Mean absolute error metric."""
        return np.mean((self.targets - self.predictions) ** 2)

    def corr_coef(self, reduce='mean', **kwargs):
        """Correlation coefficients."""
        a = self.targets
        b = self.predictions
        a = (a - np.mean(a, axis=1, keepdims=True))
        std = np.std(a, axis=1, keepdims=True)
        std[~(std > 0)] = 1
        a = a / std

        b = (b - np.mean(b, axis=1, keepdims=True))
        std = np.std(b, axis=1, keepdims=True)
        std[~(std > 0)] = 1
        b = b / std

        corr = (a * b).sum(axis=1) / a.shape[1]
        if reduce is None:
            return corr
        if isinstance(reduce, str):
            return getattr(np, reduce)(corr, **kwargs)

        return reduce(corr, **kwargs)

    def signaltonoise(self, src, signal_frame, noise_frame):
        """Signal-to-noise ratio estimation based on two frames in a seismogram."""
        data = getattr(self, src)
        return snr(data[signal_frame], data[noise_frame])

    def signaltonoise2(self, src_clean, src_noised, frame=None, mask=False):
        """Signal-to-noise ratio estimation by seismogram difference."""
        clean = getattr(self, src_clean)
        noised = getattr(self, src_noised)
        if mask and (frame is not None):
            raise ValueError("Mask or frame shoud be specified. Not both.")

        if mask:
            clean = clean[self.mask]
            noised = noised[self.mask]
        if frame:
            clean = clean[frame]
            noised = noised[frame]

        return snr(clean, noised - clean)

    def wspec(self, **kwargs):
        """Windowed spectrogram metric."""
        return np.mean(get_windowed_spectrogram_dists(self.targets, self.predictions, **kwargs))

class PickingMetrics(Metrics):
    """Class for First Break picking task metrics.

    Parameters
    ----------
    predictions : array-like
        Model predictions.
    targets : array-like
        Ground truth picking.
    gap : int (defaut=3)
        Maximum difference between prediction and target the trace considered correctly classified.
    """
    def __init__(self, targets, predictions, gap=3):
        super().__init__()
        self.targets = np.array(targets)
        self.predictions = np.array(predictions)
        self.gap = gap

    def mae(self):
        """Mean absolute error metric."""
        return np.mean(np.abs(self.targets - self.predictions))

    def accuracy(self):
        """Accuracy metric in case the task is being interpreted as classification."""
        abs_diff = np.abs(self.targets - self.predictions)
        return 100 * len(abs_diff[abs_diff < self.gap]) / len(abs_diff)

def calc_derivative_diff(ampl_diff, window=51):
    """Derivative difference metric."""
    result = measure_gain_amplitude(ampl_diff, window)
    return np.median(np.abs(np.gradient(result)))
