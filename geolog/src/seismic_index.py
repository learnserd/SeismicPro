"""Index for SeismicBatch."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..batchflow import DatasetIndex

from . import batch_tools as bt


class TraceIndex(DatasetIndex):
    """Index for individual seismic traces.

    Parameters
    ----------
    kwargs : dict
        Named argumets for ```build_df``` method.
        Can be either a set of ```dfr```, ```dfs```, ```dfx``` argumets for
        building index from SPS files, or named arguments for ```batchflow.FilesIndex```
        for building index from SEGY files.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    """
    def __init__(self, *args, index_name=None, **kwargs):
        self.meta = {}
        self.index_name = index_name
        self._idf = None
        super().__init__(*args, **kwargs)

    @property
    def tracecount(self):
        """Return a number of indexed traces."""
        return len(self._idf)

    def get_df(self, index=None, reset_index=True):
        """Return index DataFrame.

        Parameters
        ----------
        index : array-like, optional
            Subset of indices to loc from DataFrame. If None, get all the DataFrame.
        reset_index : bool, default to True
            Reset noname DataFrame index.

        Returns
        -------
        df : pandas.DataFrame
            Index DataFrame.
        """
        if index is None:
            df = self._idf
        else:
            df = self._idf.loc[index]

        if reset_index:
            return df.reset_index(drop=self.index_name is None)

        return df

    def head(self, *args, **kwargs):
        """Return the first n rows of the index DataFrame.
        
        Parameters
        ----------
        args : misc
            Positional arguments to ```pandas.DatFrame.head```.
        kwargs : dict
            Named arguments to ```pandas.DatFrame.head```.

        Returns
        -------
        First n rows of the index DataFrame.
        """
        return self._idf.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        """Return the last n rows of the index DataFrame.
       
        Parameters
        ----------
        args : misc
            Positional arguments to ```pandas.DatFrame.tail```.
        kwargs : dict
            Named arguments to ```pandas.DatFrame.tail```.

        Returns
        -------
        Last n rows of the index DataFrame.
        """
        return self._idf.tail(*args, **kwargs)

    def duplicated(self):
        """Get mask of duplicated ('FieldRecord', 'TraceNumber') pairs."""
        subset = [('FieldRecord', ''), ('TraceNumber', '')]
        return self._idf.duplicated(subset=subset)

    def drop_duplicates(self, keep='first'):
        """Drop duplicated ('FieldRecord', 'TraceNumber') pairs."""
        subset = [('FieldRecord', ''), ('TraceNumber', '')]
        df = self.get_df().drop_duplicates(subset=subset, keep=keep)
        df.set_index(self.index_name, inplace=True)
        indices = df.index.unique().sort_values()
        return type(self).from_index(index=indices, idf=df, index_name=self.index_name)

    def merge(self, x, **kwargs):
        """Merge two DataFrameIndex on common columns.

        Parameters
        ----------
        x : pandas.DataFrame
            DataFrame to merge with.
        kwargs : dict
            Named argumets to ```pandas.DataFrame.merge```.

        Returns
        -------
        df : pandas.DataFrame
            Merged DataFrame
        """
        idf = self.get_df()
        xdf = x.get_df()
        df = idf.merge(xdf, **kwargs)
        df.set_index(self.index_name, inplace=True)
        indices = df.index.unique().sort_values()
        return type(self).from_index(index=indices, idf=df, index_name=self.index_name)

    def build_index(self, index=None, idf=None, **kwargs):
        """Build index."""
        if index is not None:
            if idf is not None:
                return self.build_from_index(index, idf)
            idf = index.get_df()
            if self.index_name is not None:
                idf.set_index(self.index_name, inplace=True)

            indices = idf.index.unique().sort_values()
            self._idf = idf.loc[indices]
            return indices

        df = self.build_df(**kwargs)
        df.reset_index(drop=df.index.name is None, inplace=True)
        if self.index_name is not None:
            df.set_index(self.index_name, inplace=True)

        indices = df.index.unique().sort_values()
        self._idf = df.loc[indices]
        return indices

    def build_df(self, **kwargs):
        """Build DataFrame."""
        if 'dfx' in kwargs.keys():
            return bt.build_sps_df(**kwargs)

        return bt.build_segy_df(**kwargs)

    def build_from_index(self, index, idf):
        """Build index from another index for indices given."""
        self._idf = idf.loc[index]
        return index

    def create_subset(self, index):
        """Return a new DataFrameIndex based on the subset of indices given."""
        return type(self).from_index(index=index, idf=self._idf, index_name=self.index_name)


class SegyFilesIndex(TraceIndex):
    """Index for SEGY files.

    Parameters
    ----------
    name : str
        Name that will be associated with traces of SEGY files.
    kwargs : dict
        Named argumets for ```batchflow.FilesIndex```.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    """
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = ('file_id', kwargs.get('name'))
        super().__init__(*args, **kwargs)


class CustomIndex(TraceIndex):
    """Index for any SEGY header.

    Parameters
    ----------
    name : str
        Any segyio.TraceField keyword that will be set as index.
    kwargs : dict
        Named argumets for ```batchflow.FilesIndex````.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    """
    def __init__(self, *args, **kwargs):
        index_name = kwargs['index_name']
        if index_name is not None:
            extra_headers = kwargs['extra_headers'] if 'extra_headers' in kwargs.keys() else []
            kwargs['extra_headers'] = list(set(extra_headers + [index_name]))
        super().__init__(*args, **kwargs)


class KNNIndex(TraceIndex):
    """Index for groups of k nearest located seismic traces.

    Parameters
    ----------
    n_neighbors : int
        Group size parameter.
    kwargs : dict
        Named argumets for ```batchflow.FilesIndex````.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    """
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
            df = field_index.get_df([fid])
            data = np.stack([df['CDP_X'], df['CDP_Y']]).T
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
            _, indices = nbrs.fit(data).kneighbors(data)
            if not np.all(indices[:, 0] == np.arange(len(data))):
                raise ValueError("Faild to build KNNIndex. Duplicated CDP.")

            dfs.append(df.iloc[np.hstack(indices)])
        df = pd.concat(dfs).reset_index(drop=True)
        indices = np.repeat(np.arange(field_index.tracecount), n_neighbors)
        df['KNN'] = indices
        return df


class FieldIndex(TraceIndex):
    """Index for field records.

    Parameters
    ----------
    kwargs : dict
        Named argumets for ```build_df```` method.
        Can be either a set of ```dfr```, ```dfs```, ```dfx``` argumets for
        building index from SPS files, or named arguments for ```batchflow.FilesIndex```
        for building index from SEGY files.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    """
    def __init__(self, *args, **kwargs):
        kwargs['index_name'] = 'FieldRecord'
        super().__init__(*args, **kwargs)


class BinsIndex(TraceIndex):
    """Index for bins of CDP.

    Parameters
    ----------
    dfr : pandas.DataFrame
        SPS R file data.
    dfs : pandas.DataFrame
        SPS S file data.
    dfx : pandas.DataFrame
        SPS X file data.
    bin_size : scalar or tuple of scalars
        Grid bin size.
    origin : array-like
        Grid origin coordinates.
    phi : scalar or array-like
         Grid orientation.
    iters : int
        Maxiimal number of iterations for grid optimization algorithm.

    Attributes
    ----------
    index_name : str or tuple of str
        Name of the DataFrame index.
    meta : dict
        Metadata about index.
    """
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
