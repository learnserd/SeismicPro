"""Utils."""
import os
import glob
import functools
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker:
    """Docstring."""
    def __init__(self, ax, im, scroll_step=1, slice_names=None,
                 pts=None, axes_names=None, **kwargs):
        self.ax = ax
        self.im = im
        self.step = scroll_step
        self.slice_names = (np.arange(im.shape[-1], dtype="int")
                            if slice_names is None else slice_names)
        self.img_kwargs = kwargs

        self.ind = im.shape[-1] // 2
        self.pts = pts
        self.axes_names = ["x", "y"] if axes_names is None else axes_names

        self.update()

    def onscroll(self, event):
        """Docstring."""
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + self.step, 0, self.im.shape[-1] - 1)
        else:
            self.ind = np.clip(self.ind - self.step, 0, self.im.shape[-1] - 1)
        self.update()

    def update(self):
        """Docstring."""
        self.ax.clear()
        self.ax.imshow(self.im[:, :, self.ind], **self.img_kwargs)
        self.ax.set_title('slice %s' % self.slice_names[self.ind])
        self.ax.set_xlabel(self.axes_names[0])
        self.ax.set_ylabel(self.axes_names[1])
        self.ax.set_aspect('auto')
        self.ax.set_xlim([0, self.im.shape[1]])
        self.ax.set_ylim([self.im.shape[0], 0])
        if self.pts is not None:
            for arr in self.pts:
                arr = arr[arr[:, -1] == self.ind]
                if len(arr) == 0:
                    continue
                self.ax.scatter(arr[:, 1], arr[:, 0], alpha=0.005)


class Layouts:
    """Docstring."""
    def __init__(self):
        """Docstring."""
        self.layers = []

    def add(self, x, y, *args, **kwargs):
        """Docstring."""
        self.layers.append(dict(x=x, y=y, args=args, kwargs=kwargs))
        return self

    def show(self, labels=None, aspect='equal', figsize=None):
        """Docstring."""
        if figsize is not None:
            plt.figure(figsize=figsize)
        for layer in self.layers:
            if labels is not None:
                if 'label' not in layer['kwargs']:
                    continue
                if layer['kwargs']['label'] not in labels:
                    continue
            plt.scatter(layer['x'], layer['y'], *layer['args'], **layer['kwargs'])
        if np.any(['label' in layer['kwargs'] for layer in self.layers]):
            plt.legend()
        plt.axes().set_aspect(aspect)
        plt.show()


def get_file_by_index(path, index):
    """Docstring."""
    all_files = glob.glob(path)
    file = [f for f in all_files if int(os.path.split(f)[1].split('_')[0]) == int(index[1])]
    if len(file) != 1:
        return None
    return file[0]

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
