"""Utils."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


def get_pts(batch, grid, i, batch_size):
    """Docstring."""
    _ = batch
    pts = grid[i * batch_size: (i + 1) * batch_size]
    if len(pts) == 0:
        raise StopIteration
    return pts

def show_cube(traces, clip_value, strides=10):
    """Docstring."""
    scaler = MinMaxScaler()
    cube = np.clip(traces, -clip_value, clip_value)

    scaler.fit(cube.reshape((-1, 1)))
    cube = scaler.transform(cube.reshape((-1, 1))).reshape(cube.shape)

    ax = plt.gca(projection='3d')
    x, y = np.mgrid[0:cube.shape[0], 0:cube.shape[1]]
    ax.plot_surface(x, y, np.zeros_like(x) + cube.shape[2],
                    facecolors=np.repeat(cube[:, :, -1:], 3, axis=-1),
                    rstride=strides, cstride=strides)

    y, z = np.mgrid[0:cube.shape[1], 0:cube.shape[2]]
    ax.plot_surface(np.zeros_like(y) + 0, y, z,
                    facecolors=np.transpose(np.repeat(cube[:1, :, :], 3, axis=0), (1, 2, 0)),
                    rstride=strides, cstride=strides)

    x, z = np.mgrid[0:cube.shape[0], 0:cube.shape[2]]
    ax.plot_surface(x, np.zeros_like(x) + cube.shape[1], z,
                    facecolors=np.transpose(np.repeat(cube[:, -1:, :], 3, axis=1), (0, 2, 1)),
                    rstride=strides, cstride=strides)

    ax.plot([0, cube.shape[0]], [cube.shape[1], cube.shape[1]],
            [cube.shape[2], cube.shape[2]], c="w")
    ax.plot([0, 0], [0, cube.shape[1]], [cube.shape[2], cube.shape[2]], c="w")
    ax.plot([0, 0], [cube.shape[1], cube.shape[1]], [0, cube.shape[2]], c="w")

    ax.set_zlim([cube.shape[2], 0])
    ax.set_xlabel("i-lines")
    ax.set_ylabel("x-lines")
    ax.set_zlabel("samples")
    plt.show()

def get_file_by_index(path, index):
    """Docstring."""
    all_files = glob.glob(os.path.join(path, '*.sgy'))
    file = [f for f in all_files if int(os.path.split(f)[1].split('_')[0]) == int(index[1])]
    if len(file) != 1:
        return None
    return file[0]

def show_1d_heatmap(idf, bin_size, *args, **kwargs):
    """Docstring."""
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([i.split('/') for i in bin_counts.index])

    bindf = pd.DataFrame(bins, columns=['line', 'pos'])
    bindf['line_code'] = bindf['line'].astype('category').cat.codes + 1
    bindf = bindf.astype({'pos': 'int'})
    bindf['counts'] = bin_counts.values
    bindf = bindf.sort_values(by='line')

    brange = np.max(bindf[['line_code', 'pos']].values, axis=0)
    h = np.zeros(brange, dtype=int)
    h[bindf['line_code'].values - 1, bindf['pos'].values - 1] = bindf['counts'].values
    
    heatmap = plt.imshow(h, *args, **kwargs)
    plt.colorbar(heatmap)
    plt.yticks(np.arange(brange[0]), bindf['line'].drop_duplicates().values, fontsize=8)
    plt.xticks(np.arange(brange[1]), np.arange(brange[1]), fontsize=8)
    plt.xlabel("Bins")
    plt.ylabel("Line index")
    plt.axes().set_aspect('auto')
    plt.show()

def show_2d_heatmap(idf, bin_size, *args, **kwargs):
    """Docstring."""
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([np.array(i.split('/')).astype(int) for i in bin_counts.index])
    brange = np.max(bins, axis=0)

    h = np.zeros(brange, dtype=int)
    h[bins[:, 0] - 1, bins[:, 1] - 1] = bin_counts.values

    heatmap = plt.imshow(h.T, origin='lower', *args, **kwargs)
    plt.colorbar(heatmap) 
    plt.xlabel('x-Bins')
    plt.xlabel('y-Bins')
    plt.show()
