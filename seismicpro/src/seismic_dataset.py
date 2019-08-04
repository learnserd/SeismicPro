"""File contains seismic dataset."""
import numpy as np
from scipy.optimize import minimize

from ..batchflow import Dataset
from ..src.seismic_index import FieldIndex
from ..src.seismic_batch import SeismicBatch


class SeismicDataset(Dataset):
    """Dataset for seismic data.
    Attributes
    ----------
    sdc_params : array of ints or float with length 2
        Contains powers of speed and time for spherical divergence correction.
    """

    def __init__(self, index, batch_class=SeismicBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.sdc_params = None

    def find_sdc_params(self, src, speed, loss, time=None, started_point=None,
                        method='Powell', bounds=None, tslice=None, **kwargs):
        """ Finding an optimal parameters for correction of spherical divergence. Finding parameters
        will be saved to dataset's attribute named ```sdc_params```.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        speed : array
            Wave propagation speed depending on the depth.
        loss : callable
            Function to minimize.
        time : array, optional
           Trace time values. The default is self.meta[src]['samples'].
        started_point : array of 2
            Started values for $v_{pow}$ and $t_{pow}$.
            Default is $v_{pow}=2$ and $t_{pow}=1$.
        method : str, optional, default ```Powell```
            Minimization method, see ```scipy.optimize.minimize```.
        bounds : sequence, optional
            Sequence of (min, max) optimization bounds for each parameter. None is used to specify no bound.
            Default is ((0, 5), (0, 5)).
        tslice : slice, optional
            Lenght of loaded field.

        Returns
        -------
            : array
            Coefficients for speed and time.

        Note
        ----
        If you want to save parameters to pipeline variable use save_to argument with following
        syntax: ```save_to=V('variable name')```.
        """
        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex, not {}".format(type(self.index)))

        batch = self.create_batch(self.indices[:1]).load(src=src, components='raw', fmt='segy', tslice=tslice)
        field = batch.raw[0]
        samples = batch.meta['raw']['samples']

        bounds = ((0, 5), (0, 5)) if bounds is None else bounds
        started_point = (2, 1) if started_point is None else started_point

        time = samples if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]
        args = field, time, speed

        func = minimize(loss, started_point, args=args, method=method, bounds=bounds, **kwargs)
        self.sdc_params = func.x
        return func.x
