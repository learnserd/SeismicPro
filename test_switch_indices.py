""" Draft for switching indices functionality """
# pylint: disable=missing-docstring
import numpy as np

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
    def assert_index_type(self, index_type):
        """ insure index is of given type """
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
        new_batch = type(self)(new_index_type(self.index).create_subset(new_fields))
        new_batch.add_components(self.components, init=new_data)

        return new_batch


class CheckIndicesBatch(SeismicBatch, CheckSeismicIndicesMixin, ReindexerBatchMixin):
    pass


class ChangeIndicesDataSet(Dataset):
    """ Can store items processing status """

    class ReindexStatsEntry:

        df_columns = {
            TraceIndex: 'TRACE_SEQUENCE_FILE',
            FieldIndex: 'FieldRecord'
        }

        def __init__(self, index, old_index_type, new_index_type):
            old_col = self.df_columns[old_index_type]
            self.new_col = new_col = self.df_columns[new_index_type]

            self.imap = index.get_df()[[old_col, new_col]].droplevel(1, axis=1).set_index(old_col)
            self.new_fields = {}
            self.processed_fields = set()
            self.new_index_type = new_index_type
            self.old_index_type = old_index_type

        def add_item(self, old_idx, components, data):

            deb = self.imap.loc[old_idx + 1]
            new_item = deb[self.new_col]
            self.imap.drop(old_idx + 1, inplace=True)

            self.new_fields.setdefault(new_item, dict.fromkeys(components, []))
            for c, d in zip(components, data):
                self.new_fields[new_item][c].append(d)

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
                        res[comp].append(np.concatenate(new_data[comp]))

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
if __name__ == "__main__":
    base_path = '/media/data/Data/datasets/Metrix_QC/2_QC_Metrix_1.sgy'

    trace_index = TraceIndex(name='raw', path=base_path, extra_headers=['ShotPoint', 'offset'])

    fi = FieldIndex(trace_index)

    ds = ChangeIndicesDataSet(trace_index, CheckIndicesBatch)
    ds.initialize_reindex_stats([(TraceIndex, FieldIndex)])

    p1 = (Pipeline().load(components='raw', fmt='sgy'))

    p2 = (p1
          .assert_index_type(TraceIndex)
          .assert_traces_shape()
          .reindex(0)
          .assert_fields_shape()
          .assert_index_type(FieldIndex)
          )

    p2 = p2 << ds

    p2.run(batch_size=800, n_epochs=1, bar=True)
