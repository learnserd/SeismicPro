""" Draft for switching indices functionality """
# pylint: disable=missing-docstring
import numpy as np

import segyio

from seismicpro.batchflow import Dataset, Pipeline, action, inbatch_parallel
from seismicpro.src import SeismicBatch, FieldIndex, TraceIndex


class CheckIndicesMixin:
    """ Indices assertions"""

    @action
    @inbatch_parallel(init='indices')
    def assert_item_shape(self, index, *expected_shape):
        """ check item shape conforms with expected shape"""
        return self._assert_item_shape(index, *expected_shape)

    def _assert_item_shape(self, index, *expected_shape):
        for src in self.components:
            pos = self.get_pos(None, src, index)
            data = getattr(self, src)[pos]

            assert len(data.shape) == len(expected_shape)
            for dim, check in zip(data.shape, expected_shape):
                if callable(check):
                    assert check(dim)
                else:
                    assert dim == check

        return self

    @action
    def assert_index_type(self, index_type, deb=None):
        """ insure index is of given type """
        if deb is not None:
            if len(deb & set(self.indices)):
                print(deb & set(self.indices))

            deb.update(set(self.indices))
        if not isinstance(self.index, index_type):
            raise ValueError("Index must be {}, not {}".format(index_type, type(self.index)))
        return self


class CheckSeismicIndicesMixin(CheckIndicesMixin):
    """ Seismic indices assertions"""
    @action
    @inbatch_parallel(init='indices')
    def assert_traces_shape(self, index):
        """ check item shape conforms with trace shape"""
        return self._assert_item_shape(index, 1, lambda x: x > 1)

    @action
    @inbatch_parallel(init='indices')
    def assert_fields_shape(self, index):
        """ check item shape conforms with fields shape"""
        return self._assert_item_shape(index, lambda x: x > 1, lambda x: x > 1)


class ReindexerBatchMixin:
    """ Reindexer """

    @property
    def reindex_stats(self):
        return self.dataset.reindex_stats

    @action
    def reindex(self, self_no):
        for i in self.indices:

            data = []
            for src in self.components:
                pos = self.get_pos(None, src, i)
                data.append(getattr(self, src)[pos])

            self.reindex_stats.add_item(i, self.components, data, self_no)

        new_fields, new_data = self.reindex_stats.get_ready_idx(self.components, self_no)

        new_index_type = self.reindex_stats.new_index_type(self_no)

        new_index_big = new_index_type(self.dataset.index)
        new_batch = type(self)(new_index_big.create_subset(new_fields))
        new_batch.add_components(self.components, init=new_data)
        new_batch.dataset = self.dataset

        return new_batch


class FakeBatch(SeismicBatch):

    @inbatch_parallel(init="_init_component", target="threads")
    def _load_from_segy_file(self, index, *args, src, dst, tslice=None):
        """Load from a single segy file."""
        _ = src, args
        pos = self.get_pos(None, "indices", index)
        path = index
        trace_seq = self.index.get_df([index])[('TRACE_SEQUENCE_FILE', src)]
        if tslice is None:
            tslice = slice(None)

        with segyio.open(path, strict=False) as segyfile:
            traces = np.atleast_2d([np.ones(1000)[tslice] * i for i in
                                    np.atleast_1d(trace_seq).astype(int)])
            samples = segyfile.samples[tslice]
            interval = segyfile.bin[segyio.BinField.Interval]

        getattr(self, dst)[pos] = traces
        if index == self.indices[0]:
            self.meta[dst]['samples'] = samples
            self.meta[dst]['interval'] = interval
            self.meta[dst]['sorting'] = None

        return self

    @action
    @inbatch_parallel(init='indices')
    def assert_traces_indices(self, index):
        for src in self.components:
            pos = self.get_pos(None, src, index)
            data = getattr(self, src)[pos]

            assert int(data[0, 0]) == index + 1


class CheckIndicesBatch(FakeBatch, CheckSeismicIndicesMixin, ReindexerBatchMixin):
    pass


class ChangeIndicesDataSet(Dataset):
    """ Can store items processing status """

    class ReindexStatsEntry:

        df_columns = {
            TraceIndex: 'TRACE_SEQUENCE_FILE',
            FieldIndex: 'FieldRecord'
        }

        def __init__(self, index, old_index_type, new_index_type):
            self.old_col = old_col = self.df_columns[old_index_type]
            self.new_col = new_col = self.df_columns[new_index_type]

            df = index.get_df()
            self.imap = df[[old_col, new_col]].droplevel(1, axis=1)

            # hack
            if TraceIndex in (old_index_type, new_index_type):
                self.imap['TRACE_SEQUENCE_FILE'] = self.imap['TRACE_SEQUENCE_FILE'] - 1

            self.new_fields = {}
            self.processed_fields = set()
            self.new_index_type = new_index_type
            self.old_index_type = old_index_type

        def add_item(self, old_idx, components, data):
            old_imap_indices = self.imap[self.imap[self.old_col] == old_idx].index
            new_items = self.imap.loc[old_imap_indices][self.new_col]
            self.imap = self.imap.drop(old_imap_indices)

            need_split = len(new_items) > 1

            for i, new_item in enumerate(new_items):
                self.new_fields.setdefault(new_item,
                                           dict.fromkeys(components, {'data': [], 'order': []}))
                for c, d in zip(components, data):
                    self.new_fields[new_item][c]['data'].append(d[i].reshape(1, -1) if need_split else d)
                    self.new_fields[new_item][c]['order'].append(old_idx)

                self.processed_fields.add(new_item)

        def get_ready_idx(self, components):
            """ get fields whose traces are all processed """
            ready_items = set()
            res = dict.fromkeys(components, [])
            for i in self.processed_fields:
                if i not in self.imap[self.new_col].values:
                    ready_items.add(i)
                    new_data = self.new_fields.pop(i)
                    for comp in new_data:
                        data = np.array(new_data[comp]['data'])
                        order = np.argsort(new_data[comp]['order'])
                        res[comp].append(np.concatenate(data[order]))

            self.processed_fields -= ready_items

            return ready_items, tuple(res.values())

    class ReindexStats:
        def __init__(self):
            self.stats = []

        def init_item(self, index, old_index_type, new_index_type):
            self.stats.append(ChangeIndicesDataSet.ReindexStatsEntry(index, old_index_type, new_index_type))

        def new_index_type(self, num):
            return self.stats[num].new_index_type

        def add_item(self, old_idx, components, data, num):
            self.stats[num].add_item(old_idx, components, data)

        def get_ready_idx(self, components, num):
            return self.stats[num].get_ready_idx(components)

    def __init__(self, index, batch_class=CheckIndicesBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.reindex_stats = None

    def initialize_reindex_stats(self, index_pairs):
        self.reindex_stats = self.ReindexStats()

        for old_index_type, new_index_type in index_pairs:
            self.reindex_stats.init_item(self.index, old_index_type, new_index_type)


# pylint: disable=invalid-name
def test1():
    base_path = '/media/data/Data/datasets/Metrix_QC/2_QC_Metrix_1.sgy'

    trace_index = TraceIndex(name='raw', path=base_path, extra_headers=['ShotPoint', 'offset'])

    orig = set(trace_index.indices)

    ds = ChangeIndicesDataSet(trace_index, CheckIndicesBatch)
    ds.initialize_reindex_stats([(TraceIndex, FieldIndex), (FieldIndex, TraceIndex)])

    p1 = (Pipeline().load(components='raw', fmt='sgy'))

    s1 = set()
    s2 = set()
    s3 = set()

    p2 = (p1
          .assert_index_type(TraceIndex, s1)
          .assert_traces_shape()
          # .assert_traces_indices()
          .reindex(0)
          .assert_fields_shape()
          .assert_index_type(FieldIndex, s2)
          .reindex(1)
          .assert_index_type(TraceIndex, s3)
          .assert_traces_shape()
          .assert_traces_indices()
          )

    p2 = p2 << ds

    p2.run(batch_size=800, n_epochs=1, bar=True, shuffle=True)

    print(len(s1), len(s2), len(s3))
    print(s2)
    print(orig - set(s1))
    print(orig - set(s3))


if __name__ == "__main__":
    test1()
