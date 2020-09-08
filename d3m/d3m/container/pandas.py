import copy as copy_module
import datetime
import logging
import typing

import numpy  # type: ignore
import pandas  # type: ignore
from pandas.core.dtypes import common as pandas_common  # type: ignore

from . import list as container_list
from d3m import deprecate, exceptions
from d3m.metadata import base as metadata_base

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import lib as pyarrow_lib  # type: ignore
except ModuleNotFoundError:
    pyarrow_lib = None

__all__ = ('DataFrame',)

logger = logging.getLogger(__name__)

# This implementation is based on these guidelines:
# https://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures

D = typing.TypeVar('D', bound='DataFrame')

Data = typing.Union[typing.Sequence, typing.Mapping]


# We have to convert our container "List" to regular list because Pandas do not accept list
# subclasses. See: https://github.com/pandas-dev/pandas/issues/21226
def convert_lists(data: Data = None) -> typing.Optional[Data]:
    if isinstance(data, list) and len(data):
        if isinstance(data, container_list.List):
            data = list(data)
        if isinstance(data, list) and isinstance(data[0], container_list.List):
            data = [list(row) for row in data]

    return data


def convert_ndarray(data: Data = None) -> typing.Optional[Data]:
    """
    If ndarray has more than 2 dimensions, deeper dimensions are converted to stand-alone numpy arrays.
    """

    if isinstance(data, numpy.ndarray) and len(data.shape) > 2:
        outer_array = numpy.ndarray(shape=(data.shape[0], data.shape[1]), dtype=numpy.object)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # This retains the type, so if "data" is a container "ndarray", then also "data[i, j]" is.
                outer_array[i, j] = data[i, j]

        return outer_array

    return data


class DataFrame(pandas.DataFrame):
    """
    Extended `pandas.DataFrame` with the ``metadata`` attribute.

    Parameters
    ----------
    data:
        Anything array-like to create an instance from.
    metadata:
        Optional initial metadata for the top-level of the data frame, or top-level metadata to be updated
        if ``data`` is another instance of this data frame class.
    index:
        Index to use for resulting frame.
    columns:
        Column labels to use for resulting frame.
    dtype:
        Data type to force.
    copy:
        Copy data from inputs.
    generate_metadata:
        Automatically generate and update the metadata.
    check:
        DEPRECATED: argument ignored.
    source:
        DEPRECATED: argument ignored.
    timestamp:
        DEPRECATED: argument ignored.

    Attributes
    ----------
    metadata:
        Metadata associated with the data frame.
    """

    metadata: metadata_base.DataMetadata

    # Reversed properties.
    _metadata = ['metadata']

    @property
    def _constructor(self) -> type:
        return DataFrame

    @deprecate.arguments('source', 'timestamp', 'check', message="argument ignored")
    def __init__(self, data: Data = None, metadata: typing.Dict[str, typing.Any] = None, index: typing.Union[pandas.Index, Data] = None,
                 columns: typing.Union[pandas.Index, Data] = None, dtype: typing.Union[numpy.dtype, str, pandas_common.ExtensionDtype] = None, copy: bool = False, *,
                 generate_metadata: bool = False, check: bool = True, source: typing.Any = None, timestamp: datetime.datetime = None) -> None:
        # If not a constructor call to this exact class, then a child constructor
        # is responsible to call a pandas constructor.
        if type(self) is DataFrame:
            pandas.DataFrame.__init__(self, data=convert_ndarray(convert_lists(data)), index=index, columns=columns, dtype=dtype, copy=copy)

        # Importing here to prevent import cycle.
        from d3m import types

        if isinstance(data, types.Container):  # type: ignore
            if isinstance(data, DataFrame):
                # We made a copy, so we do not have to generate metadata.
                self.metadata: metadata_base.DataMetadata = data.metadata
            else:
                self.metadata: metadata_base.DataMetadata = data.metadata
                if generate_metadata:
                    self.metadata = self.metadata.generate(self)

            if metadata is not None:
                self.metadata: metadata_base.DataMetadata = self.metadata.update((), metadata)
        else:
            self.metadata: metadata_base.DataMetadata = metadata_base.DataMetadata(metadata)
            if generate_metadata:
                self.metadata = self.metadata.generate(self)

    def __finalize__(self: D, other: typing.Any, method: str = None, **kwargs: typing.Any) -> D:
        self = super().__finalize__(other, method, **kwargs)

        # Merge operation: using metadata of the left object.
        if method == 'merge':
            obj = other.left
        # Concat operation: using metadata of the first object.
        elif method == 'concat':
            obj = other.objs[0]
        else:
            obj = other

        if isinstance(obj, DataFrame):
            # TODO: We could adapt (if this is after a slice) metadata instead of just copying?
            self.metadata: metadata_base.DataMetadata = obj.metadata
        # "metadata" attribute should already be set in "__init__",
        # but if we got here without it, let's set it now.
        elif not hasattr(self, 'metadata'):
            self.metadata: metadata_base.DataMetadata = metadata_base.DataMetadata()

        return self

    def __getstate__(self) -> dict:
        state = super().__getstate__()

        state['metadata'] = self.metadata

        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self.metadata = state['metadata']

    def to_csv(self, path_or_buf: typing.Union[typing.IO[typing.Any], str] = None, sep: str = ',', na_rep: str = '',
               float_format: str = None, columns: typing.Sequence = None, header: typing.Union[bool, typing.Sequence[str]] = True,
               index: bool = False, **kwargs: typing.Any) -> typing.Optional[str]:
        """
        Extends `pandas.DataFrame` to provide better default method for writing DataFrames to CSV files.
        If ``header`` argument is not explicitly provided column names are derived from metadata of the DataFrame.
        By default DataFrame indices are not written.

        See Also
        --------
        `pandas.DataFrame.to_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html>`_

        Parameters
        ----------
        path_or_buf:
            File path or object, if None is provided the result is returned as a string.
        sep:
            String of length 1. Field delimiter for the output file.
        na_rep:
            Missing data representation.
        float_format:
            Format string for floating point numbers.
        columns:
            Columns to write.
        header:
            Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.
        index:
            Write row names (index).
        kwargs:
            Other arguments.
        """

        if header is True:
            header = []
            for column_index in range(len(self.columns)):
                # We use column name from the DataFrame if metadata does not have it. This allows a bit more compatibility.
                header.append(self.metadata.query_column(column_index).get('name', self.columns[column_index]))

        result = super().to_csv(path_or_buf=path_or_buf, sep=sep, na_rep=na_rep, float_format=float_format, columns=columns, header=header, index=index, **kwargs)

        # Make sure handles are flushed so that no data is lost when used with CLI file handles.
        # CLI file handles are generally used outside of a context manager which would otherwise
        # handle that.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/436
        if hasattr(path_or_buf, 'flush') and not getattr(path_or_buf, 'closed', False):
            typing.cast(typing.IO, path_or_buf).flush()

        return result

    def select_columns(self: D, columns: typing.Sequence[metadata_base.SimpleSelectorSegment], *, allow_empty_columns: bool = False) -> D:
        """
        Returns a new DataFrame with data and metadata only for given ``columns``.
        Moreover, columns are renumbered based on the position in ``columns`` list.
        Top-level metadata stays unchanged, except for updating the length of the columns dimension to
        the number of columns.

        So if the ``columns`` is ``[3, 6, 5]`` then output DataFrame will have three columns, ``[0, 1, 2]``,
        mapping data and metadata for columns ``3`` to ``0``, ``6`` to ``1`` and ``5`` to ``2``.

        This allows also duplication of columns.
        """

        if not columns and not allow_empty_columns:
            raise exceptions.InvalidArgumentValueError("No columns selected.")

        output = self.iloc[:, list(columns)]

        # We want to make sure it is a true copy.
        if output._is_view:
            output = output.copy()
        else:
            output._set_is_copy(copy=False)

        output.metadata = self.metadata.select_columns(columns, allow_empty_columns=allow_empty_columns)

        return output

    def remove_columns(self: D, column_indices: typing.Sequence[int]) -> D:
        """
        Removes columns from the DataFrame and returns one without them, together with all
        metadata for columns removed as well.

        It throws an exception if no columns would be left after removing columns.
        """

        # We are not using "drop" because we are dropping by the column index (to support columns with same name).

        columns = list(range(self.shape[1]))

        if not columns:
            raise ValueError("No columns to remove.")

        for column_index in column_indices:
            columns.remove(column_index)

        if not columns:
            raise ValueError("Removing columns would have removed the last column.")

        output = self.iloc[:, list(columns)]

        # We want to make sure it is a true copy.
        if output._is_view:
            output = output.copy()
        else:
            output._set_is_copy(copy=False)

        output.metadata = self.metadata.select_columns(columns)

        return output

    def append_columns(self: D, right: 'DataFrame', *, use_right_metadata: bool = False) -> D:
        """
        Appends all columns from ``right`` to the right of this DataFrame, together with all metadata
        of columns.

        Metadata at the top-level of ``right`` DataFrame is ignored, not merged, except if ``use_right_metadata``
        is set, in which case top-level metadata of this DataFrame is ignored and one from ``right`` is
        used instead.
        """

        outputs = pandas.concat([self, right], axis=1)
        outputs.metadata = self.metadata

        outputs.metadata = outputs.metadata.append_columns(right.metadata, use_right_metadata=use_right_metadata)

        return outputs

    def insert_columns(self: D, columns: 'DataFrame', at_column_index: int) -> D:
        """
        Inserts all columns from ``columns`` before ``at_column_index`` column in this DataFrame,
        pushing all existing columns to the right.

        E.g., ``at_column_index == 0`` means inserting ``columns`` at the beginning of this DataFrame.

        Top-level metadata of ``columns`` is ignored.
        """

        columns_length = self.shape[1]

        if at_column_index < 0:
            raise exceptions.InvalidArgumentValueError("\"at_column_index\" is smaller than 0.")
        if at_column_index > columns_length:
            raise exceptions.InvalidArgumentValueError("\"at_column_index\" is larger than the range of existing columns.")

        if at_column_index == 0:
            return columns.append_columns(self, use_right_metadata=True)

        if at_column_index == columns_length:
            return self.append_columns(columns)

        # TODO: This could probably be optimized without all the slicing and joining.

        before = self.select_columns(list(range(0, at_column_index)))
        after = self.select_columns(list(range(at_column_index, columns_length)))

        return before.append_columns(columns).append_columns(after)

    def _replace_column(self: D, column_index: int, columns: 'DataFrame', columns_column_index: int) -> D:
        # We do not use "self.iloc[:, column_index] = columns.iloc[:, columns_column_index]"
        # but use the following as a workaround.
        # See: https://github.com/pandas-dev/pandas/issues/22036
        # "self.iloc[:, [column_index]] = columns.iloc[:, [columns_column_index]]" does not work either.
        # See: https://github.com/pandas-dev/pandas/issues/22046
        output = pandas.concat([self.iloc[:, 0:column_index], columns.iloc[:, [columns_column_index]], self.iloc[:, column_index + 1:]], axis=1)
        output.metadata = output.metadata._replace_column(column_index, columns.metadata, columns_column_index)
        return output

    def replace_columns(self: D, columns: 'DataFrame', column_indices: typing.Sequence[int], *, copy: bool = True) -> D:
        """
        Replaces columns listed in ``column_indices`` with ``columns``, in order, in this DataFrame.

        ``column_indices`` and ``columns`` do not have to match in number of columns. Columns are first
        replaced in order for matching indices and columns. If then there are more ``column_indices`` than
        ``columns``, additional ``column_indices`` columns are removed. If there are more ``columns`` than
        ``column_indices`` columns, then additional ``columns`` are inserted after the last replaced column.

        If ``column_indices`` is empty, then the behavior is equivalent to calling ``append_columns``.

        Top-level metadata of ``columns`` is ignored.
        """

        # TODO: This could probably be optimized without all the slicing and joining.

        if not column_indices:
            return self.append_columns(columns)

        if copy:
            # We have to copy because "_replace" is modifying data in-place.
            outputs = copy_module.copy(self)
        else:
            outputs = self

        columns_length = columns.shape[1]
        columns_to_remove = []
        i = 0

        # This loop will run always at least once, so "column_index" will be set.
        while i < len(column_indices):
            column_index = column_indices[i]

            if i < columns_length:
                outputs = outputs._replace_column(column_index, columns, i)
            else:
                # If there are more column indices than columns in "columns", we
                # select additional columns for removal.
                columns_to_remove.append(column_index)

            i += 1

        # When there are less column indices than columns in "columns", we insert the rest after
        # the last replaced column.
        if i < columns_length:
            columns = columns.select_columns(list(range(i, columns_length)))
            # "column_index" points to the last place we inserted a column, so "+ 1" points after it.
            outputs = outputs.insert_columns(columns, column_index + 1)

        # We remove columns at the end so that we do not break and column index used before.
        # When removing columns, column indices shift.
        if columns_to_remove:
            outputs = outputs.remove_columns(columns_to_remove)

        return outputs

    def _sort_right_indices(self: 'DataFrame', right: D, indices: typing.Sequence[int], right_indices: typing.Sequence[int]) -> D:
        # We try to handle different cases.

        # We do not do anything special. We assume both indices are the same.
        if len(indices) == 1 and len(right_indices) == 1:
            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right.iloc[:, right_indices[0]]).reindex(self.iloc[:, indices[0]]).reset_index(drop=True)

        index_names = [self.metadata.query_column(index).get('name', None) for index in indices]
        right_index_names = [right.metadata.query_column(right_index).get('name', None) for right_index in right_indices]

        index_series = [self.iloc[:, index] for index in indices]
        right_index_series = [right.iloc[:, right_index] for right_index in right_indices]

        # Number match, names match, order match, things look good.
        if index_names == right_index_names:
            # We know the length is larger than 1 because otherwise the first case would match.
            assert len(indices) > 1
            assert len(indices) == len(right_indices)

            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right_index_series).reindex(index_series).reset_index(drop=True)

        sorted_index_names = sorted(index_names)
        sorted_right_index_names = sorted(right_index_names)

        # Number and names match, but not the order.
        if sorted_index_names == sorted_right_index_names:
            # We know the length is larger than 1 because otherwise the first case would match.
            assert len(indices) > 1
            assert len(indices) == len(right_indices)

            # We sort index series to be in the sorted order based on index names.
            index_series = [s for _, s in sorted(zip(index_names, index_series), key=lambda pair: pair[0])]
            right_index_series = [s for _, s in sorted(zip(right_index_names, right_index_series), key=lambda pair: pair[0])]

            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right_index_series).reindex(index_series).reset_index(drop=True)

        if len(index_series) == len(right_index_series):
            # We know the length is larger than 1 because otherwise the first case would match.
            assert len(indices) > 1

            logger.warning("Primary indices both on left and right not have same names, but they do match in number.")

            # TODO: Handle the case when not all index values exist and "reindex" fills values in: we should fill with NA relevant to the column type.
            return right.set_index(right_index_series).reindex(index_series).reset_index(drop=True)

        # It might be that there are duplicate columns on either or even both sides,
        # but that should be resolved by adding a primitive to remove duplicate columns first.
        raise ValueError("Left and right primary indices do not match in number.")

    def horizontal_concat(self: D, right: D, *, use_index: bool = True, remove_second_index: bool = True, use_right_metadata: bool = False) -> D:
        """
        Similar to ``append_columns``, but it respects primary index columns, by default.

        It has some heuristics how it tries to match up primary index columns in the case that there are
        multiple of them, but generally it aligns samples by all primary index columns.

        It is required that both inputs have the same number of samples.
        """

        self.metadata._check_same_number_of_samples(right.metadata)

        left_indices = self.metadata.get_index_columns()
        right_indices = right.metadata.get_index_columns()

        if left_indices and right_indices:
            if use_index:
                old_right_metadata = right.metadata
                right = self._sort_right_indices(right, left_indices, right_indices)
                # TODO: Reorder metadata rows as well.
                #       This should be relatively easy because we can just modify
                #       "right.metadata._current_metadata.metadata" map.
                right.metadata = old_right_metadata

            # Removing second primary key columns.
            if remove_second_index:
                right = right.remove_columns(right_indices)

        return self.append_columns(right, use_right_metadata=use_right_metadata)


def dataframe_serializer(obj: DataFrame) -> dict:
    data = {
        'metadata': obj.metadata,
        'pandas': pandas.DataFrame(obj),
    }

    if type(obj) is not DataFrame:
        data['type'] = type(obj)

    return data


def dataframe_deserializer(data: dict) -> DataFrame:
    df = data.get('type', DataFrame)(data['pandas'])
    df.metadata = data['metadata']
    return df


if pyarrow_lib is not None:
    pyarrow_lib._default_serialization_context.register_type(
        DataFrame, 'd3m.dataframe',
        custom_serializer=dataframe_serializer,
        custom_deserializer=dataframe_deserializer,
    )
