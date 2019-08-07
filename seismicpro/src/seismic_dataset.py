"""File contains seismic dataset."""
import numpy as np
from scipy.optimize import minimize

from ..batchflow import Dataset
from ..src.seismic_index import FieldIndex
from ..src.seismic_batch import SeismicBatch


class SeismicDataset(Dataset):
    """Dataset for seismic data."""

    def __init__(self, index, batch_class=SeismicBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)

    def find_sdc_params(self, components, speed, loss, indices=None, time=None, initial_point=None,
                        method='Powell', bounds=None, tslice=None, **kwargs):
        """ Finding an optimal parameters for correction of spherical divergence.

        Parameters
        ----------
        components : str or array-like, optional
            Components to load.
        speed : array
            Wave propagation speed depending on the depth.
        loss : callable
            Function to minimize.
        indices : array-like, optonal
            Which items from dataset to use in parameter estimation.
            If `None`, defaults to first element of dataset.
        time : array, optional
           Trace time values. If `None` defaults to self.meta[src]['samples'].
        initial_point : array of 2
            Started values for $v_{pow}$ and $t_{pow}$.
            If None defaults to $v_{pow}=2$ and $t_{pow}=1$.
        method : str, optional, default ```Powell```
            Minimization method, see ```scipy.optimize.minimize```.
        bounds : sequence, optional
            Sequence of (min, max) optimization bounds for each parameter.
            If `None` defaults to ((0, 5), (0, 5)).
        tslice : slice, optional
            Lenght of loaded field.

        Returns
        -------
            : array
            Coefficients for speed and time.

        Note
        ----
        To save parameters as SeismicDataset attribute use ```save_to=D('attr_name')``` (works only
        in pipeline).
        If you want to save parameters to pipeline variable use save_to argument with following
        syntax: ```save_to=V('variable_name')```.
        """
        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex, not {}".format(type(self.index)))

        if indices is None:
            indices = self.indices[:1]

        batch = self.create_batch(indices).load(components=components, fmt='segy', tslice=tslice)
        field = batch.raw[0]
        samples = batch.meta['raw']['samples']

        bounds = ((0, 5), (0, 5)) if bounds is None else bounds
        initial_point = (2, 1) if initial_point is None else initial_point

        time = samples if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]
        args = field, time, speed

        func = minimize(loss, initial_point, args=args, method=method, bounds=bounds, **kwargs)
        return func.x
