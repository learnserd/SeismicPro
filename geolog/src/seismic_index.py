"""Index for SeismicBatch."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..batchflow import DatasetIndex

from . import batch_tools as bt


class TraceIndex(DatasetIndex):
    """Base index class."""
    def __init__(self, *args, index_name=None, **kwargs):
        self.meta = {}
        self._idf = None
        self._index_name = index_name
        super().__init__(*args, **kwargs)

    def __str__(self):
        """String representation of the index DataFrame."""
        return print(self._idf)

    @property
    def name(self):
        """Return index name."""
        return self._index_name

    @property
    def tracecount(self):
        """Return a number of indexed traces."""
        return len(self._idf)

    def head(self, *args, **kwargs):
        """Return the first n rows of the index DataFrame."""
        return self._idf.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Return the last n rows of the index DataFrame."""
        return self._idf.tail(*args, **kwargs)

    def build_index(self, index=None, idf=None, **kwargs):
        """Build index."""
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            idf = index._idf # pylint: disable=protected-access
            idf = idf.reset_index(drop=(idf.index.names[0] is None))
            if self.name is not None:
                idf.set_index(self.name, inplace=True)
            self._idf = idf
            return self._idf.index.unique().sort_values()

        df = self.build_df(**kwargs)
        df.reset_index(drop=df.index.name is None, inplace=True)
        if self.name is not None:
            df.set_index(self.name, inplace=True)

        self._idf = df
        return self._idf.index.unique().sort_values()

    def build_df(self, **kwargs):
        """Build dataframe."""
        if 'dfx' in kwargs.keys():
            return bt.build_sps_df(**kwargs)
        return bt.build_segy_df(**kwargs)

    def merge(self, x, **kwargs):
        """Merge two DataFrameIndex on common columns."""
        idf = self._idf # pylint: disable=protected-access
        xdf = x._idf # pylint: disable=protected-access
        idf.reset_index(drop=idf.index.names[0] is None, inplace=True)
        xdf.reset_index(drop=xdf.index.names[0] is None, inplace=True)
        df = idf.merge(xdf, **kwargs)

        if self.name is not None:
            df.set_index(self.name, inplace=True)

        return type(self).from_index(index=df.index.unique().sort_values(), idf=df,
                                     index_name=self.name)

    def duplicated(self):
        """Duplicated ('FieldRecord', 'TraceNumber') pairs."""
        subset = [('FieldRecord', ''), ('TraceNumber', '')]
        return self._idf.duplicated(subset=subset)

    def drop_duplicates(self, keep='first'):
        """Drop duplicates ('FieldRecord', 'TraceNumber') pairs."""
        subset = [('FieldRecord', ''), ('TraceNumber', '')]
        df = (self._idf.reset_index(drop=self.name is None)
              .drop_duplicates(subset=subset, keep=keep)
              .set_index(self.name))
        return type(self).from_index(index=df.index.unique().sort_values(), idf=df, index_name=self.name)

    def sort_values(self, sort_by):
        """Sort rows."""
        self._idf.sort_values(by=sort_by, inplace=True)
        return self

    def build_from_index(self, index, idf):
        """Build index from another index for indices given."""
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """Return a new DataFrameIndex based on the subset of indices given."""
        return type(self).from_index(index=index, idf=self._idf, index_name=self.name)


class SegyFilesIndex(TraceIndex):
    """Index segy files."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = ('file_id', kwargs.get('name'))
        super().__init__(*args, **kwargs)


class CustomIndex(TraceIndex):
    """Index any segyio.TraceField attribute."""
    def __init__(self, *args, **kwargs):
        index_name = kwargs['index_name']
        if index_name is not None:
            extra_headers = kwargs['extra_headers'] if 'extra_headers' in kwargs.keys() else []
            kwargs['extra_headers'] = list(set(extra_headers + [index_name]))
        super().__init__(*args, **kwargs)


class KNNIndex(TraceIndex):
    """Index of nearest traces."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'KNN'
        super().__init__(*args, **kwargs)

    def build_df(self, n_neighbors, **kwargs):
        """Build dataframe."""
        extra_headers = kwargs['extra_headers'] if 'extra_headers' in kwargs.keys() else []
        kwargs['extra_headers'] = list(set(extra_headers + ['CDP_X', 'CDP_Y']))
        field_index = FieldIndex(**kwargs)
        dfs = []
        for fid in field_index.indices:
            df = field_index._idf.loc[fid] # pylint: disable=protected-access
            data = np.stack([df['CDP_X'], df['CDP_Y']]).T
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            _, indices = nbrs.fit(data).kneighbors(data)
            if not np.all(indices[:, 0] == np.arange(len(data))):
                raise ValueError("Faild to build KNNIndex. Duplicated CDP.")

            dfs.append(df.iloc[np.hstack(indices)])
        df = pd.concat(dfs).reset_index()
        indices = np.repeat(np.arange(field_index.tracecount), n_neighbors)
        df['KNN'] = indices
        return df


class FieldIndex(TraceIndex):
    """Index field records."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'FieldRecord'
        super().__init__(*args, **kwargs)


class BinsIndex(TraceIndex):
    """Index bins of CDP."""
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'bin_id'
        super().__init__(*args, **kwargs)

    def build_df(self, **kwargs):
        """Build dataframe."""
        df, meta = bt.make_bin_index(**kwargs)
        self.meta.update(meta)
        return df

    def show_heatmap(self, **kwargs):
        """2d histogram of CDP distribution between bins."""
        bin_size = self.meta['bin_size']
        if isinstance(bin_size, (list, tuple, np.ndarray)):
            bt.show_2d_heatmap(self._idf, **kwargs)
        else:
            bt.show_1d_heatmap(self._idf, **kwargs)
