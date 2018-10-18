from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X, step=1, slice_names=None, cmap=None):
        self.ax = ax
        self.X = X
        self.step = step
        self.slice_names = (np.arange(X.shape[-1], dtype="int")
                            if slice_names is None else slice_names)
        self.cmap = ("gray" if cmap is None else cmap)

        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap=self.cmap)
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + self.step, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - self.step, 0, self.slices - 1)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_title('slice %s' % self.slice_names[self.ind])
        self.im.axes.figure.canvas.draw()


def update_counters(batch, counters, iter_config):
    traces_shape = batch.traces[0].shape
    batch_size = iter_config["batch_size"]
    stride = iter_config["strides"]
    slice_axis = iter_config["slice_axis"]
    slice_indices = iter_config["slice_indices"]

    if counters is None:
        return [0, 0]

    slice_shape = np.delete(traces_shape, slice_axis)
    grid_size = np.prod(np.ceil(slice_shape / stride)).astype(int)
    if (counters[1] + 1) * batch_size < grid_size:
        return [counters[0], counters[1] + 1]

    if counters[0] == len(slice_indices) - 1:
        raise StopIteration

    return [counters[0] + 1, 0]


def get_start_index(batch, counters, iter_config):
    _ = batch
    return counters[1] * iter_config["batch_size"]


def get_slice_index(batch, counters, iter_config):
    _ = batch
    return iter_config["slice_indices"][counters[0]]
