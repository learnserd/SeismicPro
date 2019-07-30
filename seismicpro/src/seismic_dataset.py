"""File consists seismic dataset."""
import segyio
import numpy as np
from scipy.optimize import minimize

from ..batchflow import Dataset, NamedExpression
from ..src import FieldIndex, SegyFilesIndex

class SeismicDataset(Dataset):
    """Dataset for seismic data."""
    def _load(self, index, src, tslice=None):
        trace_seq = self.index.get_df([index])[('TRACE_SEQUENCE_FILE', src)]
        if tslice is None:
            tslice = slice(None)
        path = SegyFilesIndex(self.index, name=src).index[0]
        with segyio.open(path, strict=False) as segyfile:
            traces = np.atleast_2d([segyfile.trace[i - 1][tslice] for i in
                                    np.atleast_1d(trace_seq).astype(int)])
            samples = segyfile.samples[tslice]
        return traces, samples

    def find_correctoin_parameters(self, src, speed, time=None, loss=None, started_point=None,
                                   method='Powell', save_params_to=None, bounds=None, tslice=None):
        """
        Finding an optimal parameter for correction of spherical divergence.

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
        save_params_to : None or str, default None
            container to save parameters.
        bounds : int, default ((0, 5), (0, 5))
            Optimization bounds.
        tslice : slice, optional
            Lenght of loaded field.

        Returns
        -------
            : SeismicDataset
        """
        ix = self.indices[0]
        field, samples = self._load(ix, src, tslice)

        bounds = ((0, 5), (0, 5)) if bounds is None else bounds

        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex not {}".format(type(self.index)))

        time = samples if time is None else np.array(time, dtype=int)
        step = np.diff(time[:2])[0].astype(int)
        speed = np.array(speed, dtype=int)[::step]

        args = (field, time, speed)
        func = minimize(loss, started_point, args=args, method=method, bounds=bounds)

        v_pow, t_pow = func.x
        if isinstance(save_params_to, NamedExpression):
            save_params_to.assign((v_pow, t_pow))
        else:
            save_params_to[0] = (v_pow, t_pow)
        return self
