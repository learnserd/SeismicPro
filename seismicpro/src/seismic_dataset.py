"""File consists seismic dataset."""
import numpy as np
from scipy.optimize import minimize

from ..batchflow import Dataset
from ..src.seismic_index import FieldIndex

class SeismicDataset(Dataset):
    """Dataset for seismic data.
    Attributes
    ----------
    correction_params : array of lenght 2
        Contains powers of speed and time for spherical divergence correctoin.
    """

    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.correction_params = None

    def load_batch(self, index, src, tslice=None):
        """ Loading one element from segy file by ```index```.

        Parameters
        ----------
        index : int
            Index of loaded data in segy file.
        src : str
            The batch components to get the data from.
        tslice : slice
            Lenght of loaded field.

        Returns
        -------
            : array
            Loaded data.
            : array
            The frequency at which the measurement data is taken.
        """
        sub_ix = self.index.create_subset(np.array([index], dtype=int))
        batch = type(self)(sub_ix, self.batch_class).next_batch(1)
        batch = batch.load(src=src, fmt='segy', components=('raw'), tslice=tslice)
        data = batch.raw[0]
        samples = batch.meta['raw']['samples']
        return data, samples

    def find_correctoin_parameters(self, src, speed, time=None, loss=None, started_point=None,
                                   method='Powell', bounds=None, tslice=None):
        """ Finding an optimal parameter for correction of spherical divergence. Finding parameters
        will be saved to dataset's attribute named ```correction_params```.

        Parameters
        ----------
        src : str
            The batch components to get the data from.
        speed : array
            Wave propagation speed depending on the depth.
        time : array, optimal
           Trace time values. The default is self.meta[src]['samples'].
        loss : callable
            Function to minimize.
        started_point : array of 2
            Started values for $v_{pow}$ and $t_{pow}$.
        method : str, optional, default ```Powell```
            Minimization method, see ```scipy.optimize.minimize```.
        bounds : int, default ((0, 5), (0, 5))
            Optimization bounds.
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
        ix = self.indices[0]
        field, samples = self.load_batch(ix, src, tslice)

        bounds = ((0, 5), (0, 5)) if bounds is None else bounds

        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex not {}".format(type(self.index)))

        time = samples if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]

        args = (field, time, speed)
        func = minimize(loss, started_point, args=args, method=method, bounds=bounds)
        self.correction_params = func.x
        return func.x
