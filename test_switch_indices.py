""" Draft for switching indices functionality """

import numpy as np

from seismicpro.batchflow import Dataset, Pipeline, action, inbatch_parallel
from seismicpro.src import SeismicBatch, FieldIndex, TraceIndex

class CheckIndicesBatch(SeismicBatch):
    """ Indices assertions & switching"""

    @action
    @inbatch_parallel(init='indices')
    def assert_traces_shape(self, index):
        """ check item shape conforms with trace shape"""
        for src in self.components:
            pos = self.get_pos(None, src, index)
            data = getattr(self, src)[pos]
            assert data.shape[0] == 1
        return self

    @action
    def assert_trace_index(self):
        """ insure index is TraceIndex """
        if not isinstance(self.index, TraceIndex):
            raise ValueError("Index must be TraceIndex, not {}".format(type(self.index)))
        return self

    @action
    @inbatch_parallel(init='indices')
    def assert_fields_shape(self, index):
        """ check item shape conforms with field shape"""
        for src in self.components:
            pos = self.get_pos(None, src, index)
            data = getattr(self, src)[pos]
            assert len(data.shape) == 2
            assert data.shape[0] > 1 and data.shape[1] > 1
        return self

    @action
    def assert_field_index(self):
        """ insure index is FieldIndex """
        if not isinstance(self.index, FieldIndex):
            raise ValueError("Index must be FieldIndex, not {}".format(type(self.index)))
        return self

    @action
    def trace2field_index(self):
        """ aggregates processed traces and passes entire field when it is assembled"""
        for i in self.indices:

            data = []
            for src in self.components:
                pos = self.get_pos(None, src, i)
                data.append(getattr(self, src)[pos])

            self.dataset.add_trace(i, self.components, data)

        new_fields, new_data = self.dataset.get_ready_idx(self.components)

        new_batch = type(self)(FieldIndex(self.index).create_subset(new_fields))
        new_batch.add_components(self.components, init=new_data)

        return new_batch


class ChangeIndicesDataSet(Dataset):
    """ Can store items processing status """
    def __init__(self, index, batch_class=CheckIndicesBatch, preloaded=None, *args, **kwargs):
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)

        self.__fields_map = index.trace2field_df()
        self.__new_fields = {}
        self.__processed_fields = set()

    def add_trace(self, trace_idx, components, data):
        """ store info about processed traces """
        field = self.__fields_map.loc[trace_idx + 1].FieldRecord
        self.__new_fields.setdefault(field, dict.fromkeys(components, []))
        for c, d in zip(components, data):
            self.__new_fields[field][c].append(d)

        self.__processed_fields.add(field)
        self.__fields_map.drop(trace_idx + 1, inplace=True)

    def get_ready_idx(self, components):
        """ get fields whose traces are all processed """
        new_fields = []
        new_data = dict.fromkeys(components, [])
        for f in self.__processed_fields:
            if f not in self.__fields_map.FieldRecord.values:
                new_fields.append(f)
                field_data = self.__new_fields.pop(f)
                for comp in field_data:
                    new_data[comp].append(np.concatenate(field_data[comp]))

        self.__processed_fields -= set(new_fields)

        return new_fields, tuple(new_data.values())


class TraceIndexSwitchable(TraceIndex):
    """ Tells trace-field correspondence """
    def trace2field_df(self):
        """ Tells trace-field correspondence """
        return self._idf[['FieldRecord', 'TRACE_SEQUENCE_FILE']].droplevel(1, axis=1).set_index('TRACE_SEQUENCE_FILE')


# pylint: disable=invalid-name
if __name__ == "__main__":
    base_path = '/media/data/Data/datasets/Metrix_QC/2_QC_Metrix_1.sgy'

    trace_index = TraceIndexSwitchable(name='raw', path=base_path)

    fi = FieldIndex(trace_index)

    ds = ChangeIndicesDataSet(trace_index, CheckIndicesBatch)

    p1 = (Pipeline().load(components='raw', fmt='sgy'))

    p2 = (p1
          .assert_trace_index()
          .assert_traces_shape()
          .trace2field_index()
          .assert_fields_shape()
          .assert_field_index()
          )

    p2 = p2 << ds

    p2.run(batch_size=800, n_epochs=1, bar=True)
