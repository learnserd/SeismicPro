"""Metrics for Seismic procesing tasks."""
import numpy as np

from ..batchflow.models.metrics import Metrics

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
    