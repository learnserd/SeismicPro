"""Utils."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
from sklearn.preprocessing import MinMaxScaler

def get_file_by_index(path, index):
    """Docstring."""
    all_files = glob.glob(os.path.join(path, '*.sgy'))
    file = [f for f in all_files if int(os.path.split(f)[1].split('_')[0]) == int(index[1])]
    if len(file) != 1:
        return None
    return file[0]

class IndexTracker(object):
    """Docstring."""
    def __init__(self, ax, X, scroll_step=1, slice_names=None,
                 cmap=None, pts=None, axes_names=None):
        self.ax = ax
        self.X = X
        self.step = scroll_step
        self.slice_names = (np.arange(X.shape[-1], dtype="int")
                            if slice_names is None else slice_names)
        self.cmap = ("gray" if cmap is None else cmap)

        _, _, self.slices = X.shape
        self.ind = self.slices//2
        self.pts = pts
        self.axes_names = ["x", "y"] if axes_names is None else axes_names

        self.update()

    def onscroll(self, event):
        """Docstring."""
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + self.step, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - self.step, 0, self.slices - 1)
        self.update()

    def update(self):
        """Docstring."""
        self.ax.clear()
        self.ax.imshow(self.X[:, :, self.ind], cmap=self.cmap)
        self.ax.set_title('slice %s' % self.slice_names[self.ind])
        self.ax.set_xlabel(self.axes_names[0])
        self.ax.set_ylabel(self.axes_names[1])
        self.ax.set_aspect('auto')
        self.ax.set_xlim([0, self.X.shape[1]])
        self.ax.set_ylim([self.X.shape[0], 0])
        if self.pts is not None:
            for arr in self.pts:
                arr = arr[arr[:, -1] == self.ind]
                if len(arr) == 0:
                    continue
                self.ax.scatter(arr[:, 1], arr[:, 0], alpha=0.005)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


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
