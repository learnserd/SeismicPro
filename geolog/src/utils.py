"""Utils."""
import os
import glob
import functools
import numpy as np
import matplotlib.pyplot as plt


class IndexTracker:
    """Docstring."""
    def __init__(self, ax, frames, frame_names, scroll_step=1, **kwargs):
        self.ax = ax
        self.frames = frames
        self.step = scroll_step
        self.frame_names = frame_names
        self.img_kwargs = kwargs
        self.ind = len(frames) // 2
        self.update()

    def onscroll(self, event):
        """Docstring."""
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = np.clip(self.ind + self.step, 0, len(self.frames) - 1)
        else:
            self.ind = np.clip(self.ind - self.step, 0, len(self.frames) - 1)
        self.update()

    def update(self):
        """Docstring."""
        self.ax.clear()
        img = self.frames[self.ind]
        self.ax.imshow(img.T, **self.img_kwargs)
        self.ax.set_title('%s' % self.frame_names[self.ind])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('auto')
        self.ax.set_xlim([0, img.shape[1]])
        self.ax.set_ylim([img.shape[0], 0])


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
