import datetime
import logging
import typing

import dateutil.parser
import numpy  # type: ignore

from d3m import container, deprecate
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base

logger = logging.getLogger(__name__)

DEFAULT_DATETIME = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)


@deprecate.function(message="it should not be used anymore")
def copy_elements_metadata(source_metadata: metadata_base.Metadata, target_metadata: metadata_base.DataMetadata, from_selector: metadata_base.Selector,
                           to_selector: metadata_base.Selector = (), *, ignore_all_elements: bool = False, check: bool = True, source: typing.Any = None) -> metadata_base.DataMetadata:
    return source_metadata._copy_elements_metadata(target_metadata, list(from_selector), list(to_selector), [], ignore_all_elements)


@deprecate.function(message="use Metadata.copy_to method instead")
def copy_metadata(source_metadata: metadata_base.Metadata, target_metadata: metadata_base.DataMetadata, from_selector: metadata_base.Selector,
                  to_selector: metadata_base.Selector = (), *, ignore_all_elements: bool = False, check: bool = True, source: typing.Any = None) -> metadata_base.DataMetadata:
    return source_metadata.copy_to(target_metadata, from_selector, to_selector, ignore_all_elements=ignore_all_elements)


@deprecate.function(message="use DataFrame.select_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def select_columns(inputs: container.DataFrame, columns: typing.Sequence[metadata_base.SimpleSelectorSegment], *,
                   source: typing.Any = None) -> container.DataFrame:
    return inputs.select_columns(columns)


@deprecate.function(message="use DataMetadata.select_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def select_columns_metadata(inputs_metadata: metadata_base.DataMetadata, columns: typing.Sequence[metadata_base.SimpleSelectorSegment], *,
                            source: typing.Any = None) -> metadata_base.DataMetadata:
    return inputs_metadata.select_columns(columns)


@deprecate.function(message="use DataMetadata.list_columns_with_semantic_types method instead")
def list_columns_with_semantic_types(metadata: metadata_base.DataMetadata, semantic_types: typing.Sequence[str], *,
                                     at: metadata_base.Selector = ()) -> typing.Sequence[int]:
    return metadata.list_columns_with_semantic_types(semantic_types, at=at)


@deprecate.function(message="use DataMetadata.list_columns_with_structural_types method instead")
def list_columns_with_structural_types(metadata: metadata_base.DataMetadata, structural_types: typing.Union[typing.Callable, typing.Sequence[typing.Union[str, type]]], *,
                                       at: metadata_base.Selector = ()) -> typing.Sequence[int]:
    return metadata.list_columns_with_structural_types(structural_types, at=at)


@deprecate.function(message="use DataFrame.remove_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def remove_columns(inputs: container.DataFrame, column_indices: typing.Sequence[int], *, source: typing.Any = None) -> container.DataFrame:
    return inputs.remove_columns(column_indices)


@deprecate.function(message="use DataMetadata.remove_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def remove_columns_metadata(inputs_metadata: metadata_base.DataMetadata, column_indices: typing.Sequence[int], *, source: typing.Any = None) -> metadata_base.DataMetadata:
    return inputs_metadata.remove_columns(column_indices)


@deprecate.function(message="use DataFrame.append_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def append_columns(left: container.DataFrame, right: container.DataFrame, *, use_right_metadata: bool = False, source: typing.Any = None) -> container.DataFrame:
    return left.append_columns(right, use_right_metadata=use_right_metadata)


@deprecate.function(message="use DataMetadata.append_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def append_columns_metadata(left_metadata: metadata_base.DataMetadata, right_metadata: metadata_base.DataMetadata, use_right_metadata: bool = False, source: typing.Any = None) -> metadata_base.DataMetadata:
    return left_metadata.append_columns(right_metadata, use_right_metadata=use_right_metadata)


@deprecate.function(message="use DataFrame.insert_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def insert_columns(inputs: container.DataFrame, columns: container.DataFrame, at_column_index: int, *, source: typing.Any = None) -> container.DataFrame:
    return inputs.insert_columns(columns, at_column_index)


@deprecate.function(message="use DataMetadata.insert_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def insert_columns_metadata(inputs_metadata: metadata_base.DataMetadata, columns_metadata: metadata_base.DataMetadata, at_column_index: int, *, source: typing.Any = None) -> metadata_base.DataMetadata:
    return inputs_metadata.insert_columns(columns_metadata, at_column_index)


@deprecate.function(message="use DataFrame.replace_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def replace_columns(inputs: container.DataFrame, columns: container.DataFrame, column_indices: typing.Sequence[int], *, copy: bool = True, source: typing.Any = None) -> container.DataFrame:
    return inputs.replace_columns(columns, column_indices, copy=copy)


@deprecate.function(message="use DataMetadata.replace_columns method instead")
@deprecate.arguments('source', message="argument ignored")
def replace_columns_metadata(inputs_metadata: metadata_base.DataMetadata, columns_metadata: metadata_base.DataMetadata, column_indices: typing.Sequence[int], *, source: typing.Any = None) -> metadata_base.DataMetadata:
    return inputs_metadata.replace_columns(columns_metadata, column_indices)


@deprecate.function(message="use DataMetadata.get_index_columns method instead")
def get_index_columns(metadata: metadata_base.DataMetadata, *, at: metadata_base.Selector = ()) -> typing.Sequence[int]:
    return metadata.get_index_columns(at=at)


@deprecate.function(message="use DataFrame.horizontal_concat method instead")
@deprecate.arguments('source', message="argument ignored")
def horizontal_concat(left: container.DataFrame, right: container.DataFrame, *, use_index: bool = True,
                      remove_second_index: bool = True, use_right_metadata: bool = False, source: typing.Any = None) -> container.DataFrame:
    return left.horizontal_concat(right, use_index=use_index, remove_second_index=remove_second_index, use_right_metadata=use_right_metadata)


@deprecate.function(message="use DataMetadata.horizontal_concat method instead")
@deprecate.arguments('source', message="argument ignored")
def horizontal_concat_metadata(left_metadata: metadata_base.DataMetadata, right_metadata: metadata_base.DataMetadata, *, use_index: bool = True,
                               remove_second_index: bool = True, use_right_metadata: bool = False, source: typing.Any = None) -> metadata_base.DataMetadata:
    return left_metadata.horizontal_concat(right_metadata, use_index=use_index, remove_second_index=remove_second_index, use_right_metadata=use_right_metadata)


@deprecate.function(message="use d3m.base.utils.get_columns_to_use function instead")
def get_columns_to_use(metadata: metadata_base.DataMetadata, use_columns: typing.Sequence[int], exclude_columns: typing.Sequence[int],
                       can_use_column: typing.Callable) -> typing.Tuple[typing.List[int], typing.List[int]]:
    return base_utils.get_columns_to_use(metadata, use_columns, exclude_columns, can_use_column)


@deprecate.function(message="use d3m.base.utils.combine_columns function instead")
@deprecate.arguments('source', message="argument ignored")
def combine_columns(return_result: str, add_index_columns: bool, inputs: container.DataFrame, column_indices: typing.Sequence[int],
                    columns_list: typing.Sequence[container.DataFrame], *, source: typing.Any = None) -> container.DataFrame:
    return base_utils.combine_columns(inputs, column_indices, columns_list, return_result=return_result, add_index_columns=add_index_columns)


@deprecate.function(message="use d3m.base.utils.combine_columns_metadata function instead")
@deprecate.arguments('source', message="argument ignored")
def combine_columns_metadata(return_result: str, add_index_columns: bool, inputs_metadata: metadata_base.DataMetadata, column_indices: typing.Sequence[int],
                             columns_metadata_list: typing.Sequence[metadata_base.DataMetadata], *, source: typing.Any = None) -> metadata_base.DataMetadata:
    return base_utils.combine_columns_metadata(inputs_metadata, column_indices, columns_metadata_list, return_result=return_result, add_index_columns=add_index_columns)


@deprecate.function(message="use DataMetadata.set_table_metadata method instead")
@deprecate.arguments('source', message="argument ignored")
def set_table_metadata(inputs_metadata: metadata_base.DataMetadata, *, at: metadata_base.Selector = (), source: typing.Any = None) -> metadata_base.DataMetadata:
    return inputs_metadata.set_table_metadata(at=at)


@deprecate.function(message="use DataMetadata.get_column_index_from_column_name method instead")
def get_column_index_from_column_name(inputs_metadata: metadata_base.DataMetadata, column_name: str, *, at: metadata_base.Selector = ()) -> int:
    return inputs_metadata.get_column_index_from_column_name(column_name, at=at)


@deprecate.function(message="use Dataset.get_relations_graph method instead")
def build_relation_graph(dataset: container.Dataset) -> typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]]:
    return dataset.get_relations_graph()


@deprecate.function(message="use d3m.base.utils.get_tabular_resource function instead")
def get_tabular_resource(dataset: container.Dataset, resource_id: typing.Optional[str], *,
                         pick_entry_point: bool = True, pick_one: bool = True, has_hyperparameter: bool = True) -> typing.Tuple[str, container.DataFrame]:
    return base_utils.get_tabular_resource(dataset, resource_id, pick_entry_point=pick_entry_point, pick_one=pick_one, has_hyperparameter=has_hyperparameter)


@deprecate.function(message="use d3m.base.utils.get_tabular_resource_metadata function instead")
def get_tabular_resource_metadata(dataset_metadata: metadata_base.DataMetadata, resource_id: typing.Optional[metadata_base.SelectorSegment], *,
                                  pick_entry_point: bool = True, pick_one: bool = True) -> metadata_base.SelectorSegment:
    return base_utils.get_tabular_resource_metadata(dataset_metadata, resource_id, pick_entry_point=pick_entry_point, pick_one=pick_one)


@deprecate.function(message="use Dataset.select_rows method instead")
@deprecate.arguments('source', message="argument ignored")
def cut_dataset(dataset: container.Dataset, row_indices_to_keep: typing.Mapping[str, typing.Sequence[int]], *,
                source: typing.Any = None) -> container.Dataset:
    return dataset.select_rows(row_indices_to_keep)


def parse_datetime(value: str, *, fuzzy: bool = True) -> typing.Optional[datetime.datetime]:
    try:
        return dateutil.parser.parse(value, default=DEFAULT_DATETIME, fuzzy=fuzzy)
    except (ValueError, OverflowError, TypeError):
        return None


def parse_datetime_to_float(value: str, *, fuzzy: bool = True) -> float:
    try:
        parsed = parse_datetime(value, fuzzy=fuzzy)
        if parsed is None:
            return numpy.nan
        else:
            return parsed.timestamp()
    except (ValueError, OverflowError, TypeError):
        return numpy.nan
