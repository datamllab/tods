import collections
import copy
import os.path
import re
import typing

import numpy  # type: ignore
import pandas  # type: ignore
from pandas.io import common as pandas_common  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, params
from d3m.primitive_interfaces import base, unsupervised_learning

import common_primitives
from common_primitives import utils

__all__ = ('SimpleProfilerPrimitive',)

WHITESPACE_REGEX = re.compile(r'\s')

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    add_semantic_types: typing.Optional[typing.List[typing.List[str]]]
    remove_semantic_types: typing.Optional[typing.List[typing.List[str]]]


class Hyperparams(hyperparams_module.Hyperparams):
    detect_semantic_types = hyperparams_module.Set(
        elements=hyperparams_module.Enumeration(
            values=[
                'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
                'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
                'https://metadata.datadrivendiscovery.org/types/UniqueKey',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/Time',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                'https://metadata.datadrivendiscovery.org/types/UnknownType',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
            ],
            # Default is ignored.
            # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
            default='http://schema.org/Boolean',
        ),
        default=(
            'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
            'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
            'https://metadata.datadrivendiscovery.org/types/UniqueKey',
            'https://metadata.datadrivendiscovery.org/types/Attribute',
            'https://metadata.datadrivendiscovery.org/types/Time',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            'https://metadata.datadrivendiscovery.org/types/UnknownType',
            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
        ),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of semantic types to detect and set. One can provide a subset of supported semantic types to limit what the primitive detects.",
    )
    remove_unknown_type = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Remove \"https://metadata.datadrivendiscovery.org/types/UnknownType\" semantic type from columns on which the primitive has detected other semantic types.",
    )
    categorical_max_absolute_distinct_values = hyperparams_module.Union[typing.Union[int, None]](
        configuration=collections.OrderedDict(
            limit=hyperparams_module.Bounded[int](
                lower=1,
                upper=None,
                default=50,
            ),
            unlimited=hyperparams_module.Hyperparameter[None](
                default=None,
                description='No absolute limit on distinct values.',
            ),
        ),
        default='limit',
        description='The maximum absolute number of distinct values (all missing values as counted as one distinct value) for a column to be considered categorical.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    categorical_max_ratio_distinct_values = hyperparams_module.Bounded[float](
        lower=0,
        upper=1,
        default=0.05,
        description='The maximum ratio of distinct values (all missing values as counted as one distinct value) vs. number of rows for a column to be considered categorical.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    nan_values = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[str](''),
        default=sorted(pandas_common._NA_VALUES),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of strings to recognize as NaNs when detecting a float column.",
    )
    text_min_ratio_values_with_whitespace = hyperparams_module.Bounded[float](
        lower=0,
        upper=1,
        default=0.5,
        description='The minimum ratio of values with any whitespace (after first stripping) vs. number of rows for a column to be considered a text column.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_columns = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be detected, it is skipped.",
    )
    exclude_columns = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams_module.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should detected columns be appended, should they replace original columns, or should only detected columns be returned?",
    )
    add_index_columns = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    replace_index_columns = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Replace primary index columns even if otherwise appending columns. Applicable only if \"return_result\" is set to \"append\".",
    )


class SimpleProfilerPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which determines missing semantic types for columns and adds
    them automatically. It uses a set of hard-coded rules/heuristics to determine
    semantic types. Feel free to propose improvements.

    Besides determining column types it also determines some column roles.

    Some rules are intuitive and expected, but there are also few special behaviors
    (if not disabled by not providing a corresponding semantic type in
    ``detect_semantic_types``):

    * If a column does not have any semantic types,
      ``https://metadata.datadrivendiscovery.org/types/UnknownType`` semantic type
      is first set for the column. If any other semantic type is set later on as
      part of logic of this primitive, the
      ``https://metadata.datadrivendiscovery.org/types/UnknownType`` is removed
      (including if the column originally came with this semantic type).
    * If a column has ``https://metadata.datadrivendiscovery.org/types/SuggestedTarget``
      semantic type and no other column (even those not otherwise operated on by
      the primitive) has a semantic type
      ``https://metadata.datadrivendiscovery.org/types/TrueTarget`` is set on
      the column. This allows operation on data without a problem description.
      This is only for the first such column.
    * All other columns which are missing semantic types initially we set as
      ``https://metadata.datadrivendiscovery.org/types/Attribute``.
    * Any column with ``http://schema.org/DateTime`` semantic type is also set
      as ``https://metadata.datadrivendiscovery.org/types/Time`` semantic type.
    * ``https://metadata.datadrivendiscovery.org/types/PrimaryKey`` or
      ``https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey`` is set only
      if no other column (even those not otherwise operated on by
      the primitive) is a primary key, and set based on the column name: only
      when it is ``d3mIndex``.
    """

    __author__ = 'Louis Huang'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'e193afa1-b45e-4d29-918f-5bb1fa3b88a7',
            'version': '0.2.0',
            'name': "Determine missing semantic types for columns automatically",
            'python_path': 'd3m.primitives.schema_discovery.profiler.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:luyih@berkeley.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/simple_profiler.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_PROFILING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.SCHEMA_DISCOVERY,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._add_semantic_types: typing.List[typing.List[str]] = None
        self._remove_semantic_types: typing.List[typing.List[str]] = None
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        # The logic of detecting values tries to mirror also the logic of parsing
        # values in "ColumnParserPrimitive". One should keep them in sync.

        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        self._add_semantic_types, self._remove_semantic_types = self._fit_columns(self._training_inputs)
        self._fitted = True

        return base.CallResult(None)

    def _fit_columns(self, inputs: Inputs) -> typing.Tuple[typing.List[typing.List[str]], typing.List[typing.List[str]]]:
        true_target_columns = inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        index_columns = inputs.metadata.get_index_columns()

        # Target and index columns should be set only once, if they are set.
        has_set_target_columns = False
        has_set_index_column = False

        columns_to_use = self._get_columns(inputs.metadata)

        fitted_add_semantic_types = []
        fitted_remove_semantic_types = []

        for column_index in columns_to_use:
            input_column = inputs.select_columns([column_index])
            column_metadata = inputs.metadata.query_column(column_index)
            column_name = column_metadata.get('name', str(column_index))
            column_semantic_types = list(column_metadata.get('semantic_types', []))

            # We might be here because column has a known type, but it has "https://metadata.datadrivendiscovery.org/types/SuggestedTarget" set.
            has_unknown_type = not column_semantic_types or 'https://metadata.datadrivendiscovery.org/types/UnknownType' in column_semantic_types

            # A normalized copy of semantic types, which always includes unknown type.
            normalized_column_semantic_types = copy.copy(column_semantic_types)

            # If we are processing this column and it does not have semantic type that it has missing semantic types,
            # we first set it, to normalize the input semantic types. If we will add any other semantic type,
            # we will then remove this semantic type.
            if has_unknown_type \
                    and 'https://metadata.datadrivendiscovery.org/types/UnknownType' in self.hyperparams['detect_semantic_types'] \
                    and 'https://metadata.datadrivendiscovery.org/types/UnknownType' not in normalized_column_semantic_types:
                normalized_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/UnknownType')

            # A working copy of semantic types.
            new_column_semantic_types = copy.copy(normalized_column_semantic_types)

            if has_unknown_type:
                is_float = self._is_float(input_column)
                is_integer = self._is_integer(input_column)

                # If it looks like proper float (so not integer encoded as float), then we do not detect it as boolean.
                if self._is_boolean(input_column) \
                        and (not is_float or is_integer) \
                        and 'http://schema.org/Boolean' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Boolean' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Boolean')

                # If it looks like proper float (so not integer encoded as float), then we do not detect it as categorical.
                elif self._is_categorical(input_column) \
                        and (not is_float or is_integer) \
                        and 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in self.hyperparams['detect_semantic_types'] \
                        and 'https://metadata.datadrivendiscovery.org/types/CategoricalData' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/CategoricalData')

                elif is_integer \
                        and 'http://schema.org/Integer' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Integer' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Integer')

                elif is_float \
                        and 'http://schema.org/Float' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Float' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Float')

                elif self._is_float_vector(input_column) \
                        and 'https://metadata.datadrivendiscovery.org/types/FloatVector' in self.hyperparams['detect_semantic_types'] \
                        and 'https://metadata.datadrivendiscovery.org/types/FloatVector' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/FloatVector')

                elif self._is_datetime(input_column) \
                        and 'http://schema.org/DateTime' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/DateTime' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/DateTime')

                elif self._is_text(input_column) \
                        and 'http://schema.org/Text' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Text' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Text')

                if 'https://metadata.datadrivendiscovery.org/types/UniqueKey' in self.hyperparams['detect_semantic_types'] \
                        and self._is_unique_key(input_column) \
                        and 'http://schema.org/Text' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/UniqueKey' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/UniqueKey')

            if not true_target_columns \
                    and not has_set_target_columns \
                    and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in self.hyperparams['detect_semantic_types'] \
                    and 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in new_column_semantic_types:
                # It should not be set because there are no columns with this semantic type in whole DataFrame.
                assert 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in new_column_semantic_types
                new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                if 'https://metadata.datadrivendiscovery.org/types/Target' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                    new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
                has_set_target_columns = True

            if has_unknown_type:
                if not index_columns and not has_set_index_column:
                    if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in self.hyperparams['detect_semantic_types'] \
                            and column_name == 'd3mIndex' \
                            and 'https://metadata.datadrivendiscovery.org/types/UniqueKey' in new_column_semantic_types:
                        # It should not be set because there are no columns with this semantic type in whole DataFrame.
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types
                        new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
                        new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/UniqueKey')
                        if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                            new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
                        has_set_index_column = True
                    elif 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' in self.hyperparams['detect_semantic_types'] \
                            and column_name == 'd3mIndex':
                        assert 'https://metadata.datadrivendiscovery.org/types/UniqueKey' not in new_column_semantic_types
                        # It should not be set because there are no columns with this semantic type in whole DataFrame.
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types
                        new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey')
                        if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                            new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
                        has_set_index_column = True

                if 'https://metadata.datadrivendiscovery.org/types/Attribute' in self.hyperparams['detect_semantic_types'] \
                        and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/Attribute' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Attribute')

                if 'https://metadata.datadrivendiscovery.org/types/Time' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/DateTime' in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/Time' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Time')

                # Have we added any other semantic type besides unknown type?
                if new_column_semantic_types != normalized_column_semantic_types:
                    if self.hyperparams['remove_unknown_type'] and 'https://metadata.datadrivendiscovery.org/types/UnknownType' in new_column_semantic_types:
                        new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/UnknownType')

            new_column_semantic_types_set = set(new_column_semantic_types)
            column_semantic_types_set = set(column_semantic_types)

            fitted_add_semantic_types.append(sorted(new_column_semantic_types_set - column_semantic_types_set))
            fitted_remove_semantic_types.append(sorted(column_semantic_types_set - new_column_semantic_types_set))

        assert len(fitted_add_semantic_types) == len(columns_to_use)
        assert len(fitted_remove_semantic_types) == len(columns_to_use)

        return fitted_add_semantic_types, fitted_remove_semantic_types

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        assert self._add_semantic_types is not None
        assert self._remove_semantic_types is not None

        columns_to_use, output_columns = self._produce_columns(inputs, self._add_semantic_types, self._remove_semantic_types)

        if self.hyperparams['replace_index_columns'] and self.hyperparams['return_result'] == 'append':
            assert len(columns_to_use) == len(output_columns)

            index_columns = inputs.metadata.get_index_columns()

            index_columns_to_use = []
            other_columns_to_use = []
            index_output_columns = []
            other_output_columns = []
            for column_to_use, output_column in zip(columns_to_use, output_columns):
                if column_to_use in index_columns:
                    index_columns_to_use.append(column_to_use)
                    index_output_columns.append(output_column)
                else:
                    other_columns_to_use.append(column_to_use)
                    other_output_columns.append(output_column)

            outputs = base_utils.combine_columns(inputs, index_columns_to_use, index_output_columns, return_result='replace', add_index_columns=self.hyperparams['add_index_columns'])
            outputs = base_utils.combine_columns(outputs, other_columns_to_use, other_output_columns, return_result='append', add_index_columns=self.hyperparams['add_index_columns'])
        else:
            outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query_column(column_index)

        semantic_types = column_metadata.get('semantic_types', [])

        # We detect only on columns which have no semantic types or
        # where it is explicitly set as unknown.
        if not semantic_types or 'https://metadata.datadrivendiscovery.org/types/UnknownType' in semantic_types:
            return True

        # A special case to handle setting "https://metadata.datadrivendiscovery.org/types/TrueTarget".
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types:
            return True

        return False

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being parsed.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can parsed. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _produce_columns(
        self, inputs: Inputs,
        add_semantic_types: typing.List[typing.List[str]],
        remove_semantic_types: typing.List[typing.List[str]],
    ) -> typing.Tuple[typing.List[int], typing.List[Outputs]]:
        columns_to_use = self._get_columns(inputs.metadata)

        assert len(add_semantic_types) == len(remove_semantic_types), (len(add_semantic_types), len(remove_semantic_types))

        if len(columns_to_use) != len(add_semantic_types):
            raise exceptions.InvalidStateError("Producing on a different number of columns than fitting.")

        output_columns = []

        for column_index, column_add_semantic_types, column_remove_semantic_types in zip(columns_to_use, add_semantic_types, remove_semantic_types):
            output_column = inputs.select_columns([column_index])

            for remove_semantic_type in column_remove_semantic_types:
                output_column.metadata = output_column.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 0), remove_semantic_type)
            for add_semantic_type in column_add_semantic_types:
                output_column.metadata = output_column.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), add_semantic_type)

            output_columns.append(output_column)

        assert len(output_columns) == len(columns_to_use)

        return columns_to_use, output_columns

    def _is_boolean(self, input_column: container.DataFrame) -> bool:
        # If there are less than 3 rows, we do no detect it to be boolean ever.
        if len(input_column) < 3:
            return False

        # Or it should already be boolean dtype.
        if input_column.dtypes.iloc[0].kind == 'b':
            return True

        # Are there only two categorical values (after striping string,
        # if there are string values). Missing values (empty strings) are not counted
        # towards this, so they are not stored in "values_set".
        values_set: typing.Set[typing.Any] = set()
        for value in input_column.iloc[:, 0]:
            if value == "" or pandas.isna(value):
                continue

            if isinstance(value, str):
                value = value.strip()

            values_set.add(value)

            if len(values_set) > 2:
                return False

        assert len(values_set) <= 2

        # There should be at least one row not NaN. This prevents a degenerate case
        # where we would mark a column of no rows or just NaNs as boolean column.
        if values_set:
            return True

        return False

    # TODO: What to do when there are very little number of rows?
    #       Like 10? And 3 distinct values. This should still be seen as categorical?
    # TODO: Optimize. We od not have to compute all counts.
    #       But just to cross the limit to be able to return False.
    def _is_categorical(self, input_column: container.DataFrame) -> bool:
        # We first count all the values. We do not use "value_counts" so that we can
        # strip strings if they are strings when counting. We also put all missing values
        # as one value.
        missing_values_count = 0
        value_counts: typing.Dict[typing.Any, int] = collections.defaultdict(int)
        for value in input_column.iloc[:, 0]:
            if value == "" or pandas.isna(value):
                missing_values_count += 1
                continue

            if isinstance(value, str):
                value = value.strip()

            value_counts[value] += 1

        input_column_rows = len(input_column)
        all_distinct_values = bool(missing_values_count) + len(value_counts)

        # There should be at least one row not NaN. This prevents a degenerate case
        # where we would mark a column of no rows or just NaNs as categorical column.
        # (Otherwise we also get division by zero below.)
        if not input_column_rows:
            return False

        # Check the absolute limit.
        if self.hyperparams['categorical_max_absolute_distinct_values'] is not None \
                and all_distinct_values > self.hyperparams['categorical_max_absolute_distinct_values']:
            return False

        # Check the relative limit.
        if (all_distinct_values / input_column_rows) > self.hyperparams['categorical_max_ratio_distinct_values']:
            return False

        return True

    def _is_integer(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as integer column.
        if not len(column_values):
            return False

        # Or it should already be integer dtype.
        if input_column.dtypes.iloc[0].kind in ['i', 'u']:
            return True

        # Or is of float dtype which have all values in fact integers.
        if input_column.dtypes.iloc[0].kind == 'f':
            # If all values are or integers or NaN values.
            column_values = column_values.dropna()

            # There should be at least one row not NaN. This prevents a degenerate case
            # where we would mark a column of just NaNs as integer column.
            if len(column_values) and all(v.is_integer() for v in column_values):
                return True

            return False

        not_nan_exists = False

        # Or it should be strings (stripped) which all convert to an integer.
        for value in column_values:
            if not isinstance(value, str):
                return False

            value = value.strip()

            try:
                int(value)
                not_nan_exists = True
                continue
            except ValueError:
                pass

            try:
                # Maybe it is an int represented as a float. Let's try this.
                value = float(value)
            except ValueError:
                # No luck.
                return False

            if pandas.isna(value):
                continue

            if value.is_integer():
                not_nan_exists = True
                continue

            return False

        # There should be at least one row not NaN. This prevents a degenerate case
        # where we would mark a column of just NaNs as integer column.
        if not_nan_exists:
            return True

        return False

    def _is_float(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as float column.
        if not len(column_values):
            return False

        # Or it should already be float dtype. It is OK if there are just NaNs in this case.
        if input_column.dtypes.iloc[0].kind in ['f', 'c']:
            return True

        # Or it should be strings (stripped) which all convert to a float or a nan/missing value.
        for value in column_values:
            # TODO: Should we just look at structural type in metadata? Instead of spending time checking every value?
            if not isinstance(value, str):
                return False

            value = value.strip()

            try:
                value = float(value)
                continue
            except ValueError:
                pass

            # We allow some string values to exist. When parsing they are parsed as float NaNs.
            if value in self.hyperparams['nan_values']:
                continue

            return False

        # We do mark a column of all NaNs as float column. This includes marking a column of just
        # empty strings as float column. It has to be something, so a float column seems reasonable.
        return True

    def _is_float_vector(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as float vector column.
        if not len(column_values):
            return False

        try:
            structural_type = input_column.metadata.query_column_field(0, 'structural_type')
        except KeyError:
            structural_type = None

        # Or it is already parsed as 1d ndarray of floats or ints.
        if structural_type is not None and d3m_utils.is_subclass(structural_type, container.ndarray):
            for value in column_values:
                # It has to be a vector ndarray.
                if len(value.shape) != 1:
                    return False

                # With floats or ints.
                if value.dtype.kind not in ['f', 'i']:
                    return False

            return True

        vector_exists = False

        # Or it is a string which can be split by "," and each be parsed as float (without missing values).
        # We are pretty strict here because we are assuming this was generated programmatically.
        for value in column_values:
            # TODO: Should we just look at structural type in metadata? Instead of spending time checking every value?
            #       But what if "structural_type" is None? Do we want to support that? We probably should not.
            if not isinstance(value, str):
                return False

            values = value.split(',')

            if not values:
                continue

            for value in values:
                try:
                    value = float(value)
                except ValueError:
                    return False

                if pandas.isna(value):
                    return False

                vector_exists = True

        # There should be at least one row with non-empty vector. This prevents a degenerate case
        # where we would mark a column of empty strings as float vector column.
        if vector_exists:
            return True

        return False

    def _is_datetime(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as a datetime column.
        if not len(column_values):
            return False

        # Or it should already be datetime dtype.
        if input_column.dtypes.iloc[0].kind == 'M':
            return True

        datetime_exists = False

        # Or the value is not-a-datetime value, or it can be parsed as a datetime.
        for value in column_values:
            # TODO: Should we just look at structural type in metadata? Instead of spending time checking every value?
            if not isinstance(value, str):
                return False

            # TODO: Allow any other not-a-datetime value? Like string version of Panda's NaT?
            if value == "":
                continue

            if numpy.isnan(utils.parse_datetime_to_float(value)):
                return False
            else:
                datetime_exists = True

        # There should be at least one row not NaN. This prevents a degenerate case
        # where we would mark a column of empty strings as datetime column.
        if datetime_exists:
            return True

        return False

    # TODO: Optimize. We od not have to check all values.
    #       If we cross the limit where we already have more than ratio values, we can return True.
    def _is_text(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as a text column.
        # (Otherwise we also get division by zero below.)
        if not len(column_values):
            return False

        values_with_whitespace = 0

        # It has to be structural type string and at least 50 % of rows should have a whitespace
        # in them after the value has been stripped.
        for value in column_values:
            # TODO: Should we just look at structural type in metadata? Instead of spending time checking every value?
            if not isinstance(value, str):
                return False

            value = value.strip()

            if WHITESPACE_REGEX.search(value):
                values_with_whitespace += 1

        if (values_with_whitespace / len(column_values)) < self.hyperparams['text_min_ratio_values_with_whitespace']:
            return False

        return True

    def _is_unique_key(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as a unique key column.
        # (Otherwise we also get division by zero below.)
        if not len(column_values):
            return False

        # Here we look at every value as-is. Even empty strings and other missing/nan values.
        if any(input_column.duplicated()):
            return False

        return True

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                add_semantic_types=None,
                remove_semantic_types=None,
            )

        return Params(
            add_semantic_types=self._add_semantic_types,
            remove_semantic_types=self._remove_semantic_types,
        )

    def set_params(self, *, params: Params) -> None:
        self._add_semantic_types = params['add_semantic_types']
        self._remove_semantic_types = params['remove_semantic_types']
        self._fitted = all(param is not None for param in params.values())
