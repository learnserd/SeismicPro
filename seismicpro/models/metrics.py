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

class FieldMetrics(Metrics):
    """Class for seismic field record metrics.
    """
    def __init__(self, targets, predictions):
        super().__init__()
        self.targets = targets
        self.predictions = predictions

    def iou(self):
        """Intersection-over-union metric."""
        a = self.targets.astype(float)
        b = self.predictions.astype(float)
        return 2 * np.sum(a * b) / np.sum(a + b)

    def mae(self):
        """Mean absolute error metric."""
        return np.mean(abs(self.targets - self.predictions))

    def corr_coef(self, reduce='mean', **kwargs):
        """Correlation coeffitients."""
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
