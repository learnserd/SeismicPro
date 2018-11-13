from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X, scroll_step=1, slice_names=None,
                 cmap=None, pts=None, axes_names=None):
        self.ax = ax
        self.X = X
        self.step = scroll_step
        self.slice_names = (np.arange(X.shape[-1], dtype="int")
                            if slice_names is None else slice_names)
        self.cmap = ("gray" if cmap is None else cmap)

        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.pts = pts
        self.axes_names = ["x", "y"] if axes_names is None else axes_names

        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + self.step, 0, self.slices - 1)
        else:
            self.ind = np.clip(self.ind - self.step, 0, self.slices - 1)
        self.update()

    def update(self):
        self.ax.clear()
        self.ax.imshow(self.X[:, :, self.ind], cmap=self.cmap)
        self.ax.set_title('slice %s' % self.slice_names[self.ind])
        self.ax.set_xlabel(self.axes_names[0])
        self.ax.set_ylabel(self.axes_names[1])
        self.ax.set_aspect('auto')
        self.ax.set_xlim([0, self.X.shape[1]])
        self.ax.set_ylim([self.X.shape[0], 0])
        if self.pts is not None:
            pcolors = np.linspace(0, 1, len(self.pts))
            for i, arr in enumerate(self.pts):
                arr = arr[arr[:, -1] == self.ind]
                if len(arr) == 0:
                    continue
                self.ax.scatter(arr[:, 1], arr[:, 0], alpha=0.005)


def get_pts(batch, grid, i, batch_size):
    pts = grid[i * batch_size: (i + 1) * batch_size]
    if len(pts) == 0:
        raise StopIteration
    return pts
