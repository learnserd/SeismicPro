"""Seismic batch."""
import glob
import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
from numba import njit

from dataset import (action, inbatch_parallel, Batch,
                     FilesIndex, DatasetIndex,
                     ImagesBatch, any_action_failed)

from field_index import FieldIndex
from utils import IndexTracker


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


class SeismicBatch(Batch):
    """Docstring."""
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded=preloaded)
        if preloaded is None:
            self.traces = np.array([None] * len(self.index))
            self.annotation = np.array([None] * len(self.index))
            self.meta = np.array([None] * len(self.index))

    @action
    @inbatch_parallel(init="indices", target="threads")
    def to_2d(self, index):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces = self.traces[pos]
        if traces is None or len(traces) == 0:
            return
        try:
            traces_2d = np.vstack(traces)
        except ValueError:
            shape = (len(traces), max([len(t) for t in traces]))
            traces_2d = np.zeros(shape)
            for i, arr in enumerate(traces):
                traces_2d[i, :len(arr)] = arr
        self.traces[pos] = traces_2d

    @action
    def load(self, src=None, path=None, fmt=None, *args, **kwargs):
        """Docstring."""
        if isinstance(self.index, FilesIndex) or (src is not None):
            return self._load_from_paths(src=src, fmt=fmt, *args, **kwargs)
        elif isinstance(self.index, FieldIndex):
            return self._load_from_traces(path=path, fmt=fmt, *args, **kwargs)
        else:
            raise NotImplementedError("Unknown index type.")

    def _load_from_traces(self, path=None, fmt=None, sort_by='r2',
                          get_file_by_index=None, skip_channels=0):
        """Docstring."""
        src = []
        channels = []
        pos = []

        idf = self.index._idf
        idf['_pos'] = np.arange(len(idf))

        for index, group in idf.groupby(['tape', 'xid']):
            file = get_file_by_index(path, index)
            if file is not None:
                src.append(file)
                channels.append(group['channel'].values)
                pos.extend(group['_pos'].values)

        if not src:
            return self

        batch = (type(self)(DatasetIndex(np.arange(len(src))))
                 .load(src=src, fmt=fmt, channels=channels, skip_channels=skip_channels))

        all_traces = np.array([t for item in batch.traces for t in item] + [None])[:-1]

        idf['_trace'] = np.nan
        idf.iloc[pos, idf.columns.get_loc('_trace')] = all_traces

        for index, group in idf.groupby(level=0):
            ipos = self.get_pos(None, "indices", index)
            group = group.dropna(axis=0).sort_values(by=sort_by)
            self.traces[ipos] = group['_trace'].values
            self.meta[ipos] = dict(sorting=sort_by,
                                   sht_depth=group['sht_depth'].values if 'sht_depth' in group.columns else None,
                                   uphole=group['uphole'].values if 'uphole' in group.columns else None,
                                   z=group['z_s'].values if 'z_s' in group.columns else None)

        idf.drop(labels=['_pos', '_trace'], axis=1, inplace=True)

        return self


    @inbatch_parallel(init="indices", target="threads")
    def _load_from_paths(self, index, src=None, fmt=None, channels=None, skip_channels=0):
        """Docstring."""
        if src is not None:
            path = src[index]
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source is not specified")
        pos = self.get_pos(None, "indices", index)
        if fmt == "segy":
            with segyio.open(path, strict=False) as sf:
                if (sf.sorting is not None) and (channels is None):
                    self.traces[pos] = segyio.tools.cube(sf)
                else:
                    if channels is None:
                        self.traces[pos] = sf.trace.raw[:][slice(skip_channels, None)]
                    else:
                        self.traces[pos] = sf.trace.raw[:][channels[index] - 1 + skip_channels]
                self.meta[pos] = segyio.tools.metadata(sf).__dict__
        elif fmt == "pts":
            pdir = os.path.split(path)[0] + '/*.pts'
            files = glob.glob(pdir)
            self.annotation[pos] = []
            for f in files:
                self.annotation[pos].append(np.loadtxt(f))
            self.annotation[pos] = np.array(self.annotation[pos])
        else:
            raise NotImplementedError("Unknown file format.")

    @action
    @inbatch_parallel(init="indices", target="threads")
    def sort_traces(self, index, sort_by):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        if isinstance(self.index, FieldIndex):
            idf = self.index._idf.loc[index]
        else:
            raise ValueError("Sorting is not supported for this Index class")
        order = np.argsort(idf[sort_by].values)
        self.traces[pos] = self.traces[pos][order]
        self.meta[pos]['sorting'] = sort_by

    @action
    @inbatch_parallel(init="indices", target="threads")
    def filter_annotations(self, index, ann_names, mode):
        """Docstring."""
        if isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")
        pdir = os.path.split(path)[0] + '/*.pts'
        files = np.array([os.path.split(p)[1] for p in glob.glob(pdir)])
        pos = self.get_pos(None, "indices", index)
        indices = []
        for ann in ann_names:
            i = np.where(files == ann)[0]
            if len(i):
                indices.append(i[0])
        if mode == "drop":
            self.annotation[pos] = np.delete(self.annotation[pos], indices)
        elif mode == "keep":
            self.annotation[pos] = self.annotation[pos][indices]
        else:
            raise ValueError("Unknown filter mode")
        return self

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list.
        """
        if any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    def _assemble_crops(self, results, *args, **kwargs):
        """Concatenate results of different workers.
        """
        _ = args, kwargs
        self._reraise_exceptions(results)
        crops, labels = list(zip(*results))

        crops = np.vstack(crops)
        labels = np.hstack(labels)

        return ImagesBatch(DatasetIndex(np.arange(len(crops))),
                           preloaded=(crops, labels, np.zeros(len(crops))))

    @action
    @inbatch_parallel(init="indices", post="_assemble_crops", target="threads")
    def sample_crops(self, index, size, origin, n_crops=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]

        if type(origin) in [list, tuple, np.ndarray]:
            labels = np.array([None] * len(origin))
            sampled_pts = np.array(origin)

        elif origin == "random_annotated_unbalanced":
            stacked_pts = np.vstack(pts)[:, :3]
            pos = np.random.randint(0, len(stacked_pts), size=n_crops)
            sampled_pts = pts_to_indices(stacked_pts[pos], meta)
            labels = np.repeat(np.arange(len(pts)), [len(p) for p in pts])[pos]

        elif origin == "random_annotated_balanced":
            labels = np.random.choice(np.arange(len(pts)), size=n_crops)
            sampled_pts = np.zeros((n_crops, 3))
            for i, arr in enumerate(pts):
                mask = np.where(labels == i)
                pos = np.random.randint(0, len(arr), size=len(mask[0]))
                sampled_pts[mask] = arr[pos, :3]
            sampled_pts = pts_to_indices(sampled_pts, meta)
        else:
            raise ValueError("Unknown sampling mode")

        if isinstance(size, int):
            size = tuple([size] * traces.ndim)

        if traces.ndim == 2:
            size = size + (1,)

        crops = (nj_sample_crops(np.atleast_3d(traces), sampled_pts, size)
                 .reshape((-1,) + size[:traces.ndim]))

        return [crops, labels]

    def slice_tracker(self, index, axis, scroll_step=1, show_pts=False, cmap=None):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = self.traces[pos], self.annotation[pos], self.meta[pos]

        order = np.hstack((np.delete(np.arange(traces.ndim), axis)[::-1], axis))
        axes_names = np.delete(["i-lines", "x-lines", "samples"], axis)

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(axes_names[0])
        ax.set_ylabel(axes_names[1])

        if show_pts:
            ipts = [pts_to_indices(arr[:, :3], meta)[:, order] for arr in pts]
        else:
            ipts = None
        tracker = IndexTracker(ax, np.transpose(traces, order), cmap=cmap,
                               scroll_step=scroll_step, pts=ipts, axes_names=axes_names)
        return fig, tracker

    def show_slice(self, index, axis=-1, offset=0, show_pts=False, **kwargs):
        """Docstring."""
        pos = self.get_pos(None, "indices", index)
        traces, pts, meta = np.atleast_3d(self.traces[pos]), self.annotation[pos], self.meta[pos]
        ix = [slice(None)] * traces.ndim
        ix[axis] = offset
        ax = np.delete(np.arange(3), axis)

        if meta["sorting"] == 2:
            axes_names = np.delete(["i-lines", "x-lines", "samples"], axis)
        elif meta["sorting"] == 1:
            axes_names = np.delete(["x-lines", "i-lines", "samples"], axis)
        else:
            axes_names = np.delete(["x", "y", "z"], axis)

        plt.imshow(traces[ix].T, **kwargs)

        if show_pts:
            ipts = np.array([pts_to_indices(p[:, :3], meta) for p in pts])
            for arr in ipts:
                arr = arr[arr[:, axis] == offset]
                plt.scatter(arr[:, ax[0]], arr[:, ax[1]], alpha=0.005)

        plt.ylabel(axes_names[1])
        plt.xlabel(axes_names[0])
        plt.axis('auto')
        plt.xlim([0, traces.shape[ax[0]]])
        plt.ylim([traces.shape[ax[1]], 0])
        plt.show()
