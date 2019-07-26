"""Metrics for Seismic procesing tasks."""
import numpy as np

from ..batchflow.models.metrics import Metrics

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

    def corrcoef(self, reduce='mean', **kwargs):
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
    