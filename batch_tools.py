"""Seismic batch."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from sklearn.preprocessing import MinMaxScaler

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

def show_1d_heatmap(idf, *args, **kwargs):
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

def show_2d_heatmap(idf, *args, **kwargs):
    """Docstring."""
    bin_counts = idf.groupby(level=[0]).size()
    bins = np.array([np.array(i.split('/')).astype(int) for i in bin_counts.index])
    brange = np.max(bins, axis=0)

    h = np.zeros(brange, dtype=int)
    h[bins[:, 0] - 1, bins[:, 1] - 1] = bin_counts.values

    heatmap = plt.imshow(h.T, origin='lower', *args, **kwargs)
    plt.colorbar(heatmap)
    plt.xlabel('x-Bins')
    plt.ylabel('y-Bins')
    plt.show()

@njit(nogil=True)
def nj_sample_crops(traces, pts, size):
    """Docstring."""
    res = np.zeros((len(pts), ) + size, dtype=traces.dtype)
    asize = np.array(size)
    offset = asize // 2
    start = np.zeros(3, dtype=pts.dtype)
    t_stop = np.zeros(3, dtype=pts.dtype)
    c_stop = np.zeros(3, dtype=pts.dtype)
    for i, p in enumerate(pts):
        start[:p.size] = p - offset[:p.size]
        t_stop[:p.size] = p + asize[:p.size] - offset[:p.size]

        t_start = np.maximum(start, 0)
        step = (np.minimum(p + asize[:p.size] - offset[:p.size],
                           np.array(traces.shape)[:p.size]) - t_start[:p.size])

        c_start = np.maximum(-start, 0)
        c_stop[:p.size] = c_start[:p.size] + step

        res[i][c_start[0]: c_stop[0], c_start[1]: c_stop[1], c_start[2]: c_stop[2]] =\
            traces[t_start[0]: t_stop[0], t_start[1]: t_stop[1], t_start[2]: t_stop[2]]

    return res

def pts_to_indices(pts, meta):
    """Docstring."""
    starts = np.array([meta['ilines'][0], meta['xlines'][0], meta['samples'][0]])
    steps = np.array([meta['ilines'][1] - meta['ilines'][0],
                      meta['xlines'][1] - meta['xlines'][0],
                      meta['samples'][1] - meta['samples'][0]])
    return ((pts - starts) / steps).astype(int)
