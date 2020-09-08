import os
import typing
from typing import Any, Dict, List, Tuple

import d3m.metadata.base as metadata_module
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from d3m import exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.container.pandas import DataFrame
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

import common_primitives

Inputs = DataFrame
Outputs = DataFrame


class Params(params.Params):
    categories: Dict[int, np.ndarray]
    fitted: bool


class Hyperparams(hyperparams.Hyperparams):
    separator = hyperparams.Hyperparameter[str](
        default='.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Separator separates additional identifier and original column name",
    )
    prefix = hyperparams.Hyperparameter[str](
        default='col',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Separator separates additional identifier and original column name",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of inputs column indices to force primitive to operate on. If any specified column cannot "
                    "be used, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of inputs column indices to not operate on. Applicable only if \"use_columns\" is not "
                    "provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='new',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed "
                    "columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    handle_unseen = hyperparams.Enumeration(
        values=['error', 'ignore', 'column'],
        default='ignore',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="error: throw exception when unknown value observed"
                    "ignore: ignore unseen values"
                    "auto: put unseen values in extra column and mark the cell as 1"
    )
    handle_missing_value = hyperparams.Enumeration(
        values=['error', 'ignore', 'column'],
        default='ignore',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Options for dealing with missing values.'
                    'error: throw exceptions when missing values encountered.'
                    'ignore: ignore any missing value.'
                    'column: add one column for missing value.'
    )
    # TODO hyperparams.Hyperparameter[typing.Set[Any]] doesn't work?
    missing_values = hyperparams.Hyperparameter[typing.AbstractSet[Any]](
        default={np.NaN, None, ''},
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Values indicate the data is a missing other than 'None' and 'np.NaN'",
    )
    encode_target_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether to encode target column",
    )


class OneHotMakerPrimitive(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Attempts to detect discrete values in data and convert these to a
    one-hot embedding.
    """
    _unseen_column_name: str = 'Unseen'
    _missing_column_name: str = 'Missing'

    metadata = metadata_module.PrimitiveMetadata({
        'id': 'eaec420d-46eb-4ddf-a2cd-b8097345ff3e',
        'version': '0.3.0',
        'name': 'One-hot maker',
        'keywords': ['data processing', 'one-hot'],
        'source': {
            'name': common_primitives.__author__,
            'contact': 'mailto:lin.yang@tamu.edu',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/one_hot_maker.py',
                'https://gitlab.com/datadrivendiscovery/common-primitives.git',
            ],
        },
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)))
        }],
        'python_path': 'd3m.primitives.data_preprocessing.one_hot_encoder.MakerCommon',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.ENCODE_ONE_HOT,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._training_inputs: Inputs = None
        self._categories: Dict[int, np.array] = {}
        self._fitted: bool = False

        # record unseen row index and column name
        self._unseen: List[Tuple[int, str]] = []
        self._missing: Dict[str, List[int]] = {}

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        column_indices = self._get_columns(self._training_inputs.metadata)
        for i in column_indices:
            self._categories[i] = self._fit_categories(self._training_inputs.iloc[:, i])
        self._fitted = True

        return CallResult(None)

    def _fit_categories(self, column: pd.Series) -> np.array:
        # generates sorted unique value
        missing_value_mask = self._get_missing_value_mask(column)
        if self.hyperparams['handle_missing_value'] == 'error':
            if missing_value_mask.any():
                raise exceptions.MissingValueError('Missing value in categorical data')
        _categories = np.unique(column[~missing_value_mask])
        return _categories

    def produce(self, *, inputs: Inputs,
                timeout: float = None,
                iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")
        selected_inputs, columns_to_use = self._select_columns(inputs)
        if len(selected_inputs.columns[selected_inputs.columns.duplicated()].unique()):
            raise exceptions.ColumnNameError('Duplicated column name')
        # TODO check if input has the same column as input in fit stage

        outputs = []
        for i in columns_to_use:
            input = inputs.iloc[:, i]
            onehot_result = self._produce_onehot_columns(i, input)
            column_names = self._produce_onehot_column_names(i, inputs.metadata, input.name)
            output = DataFrame(onehot_result, columns=column_names, generate_metadata=False)
            self._produce_onehot_metadata(inputs, output, i)
            outputs.append(output)

        outputs = base_utils.combine_columns(inputs, columns_to_use, outputs,
                                             return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'])
        return CallResult(outputs)

    def _produce_onehot_columns(self, column_index: int, column: pd.Series) -> np.ndarray:
        category = self._categories[column_index]
        column_count = len(category)
        row_count = len(category) + 2
        handle_missing_value = self.hyperparams['handle_missing_value']
        handle_unseen = self.hyperparams['handle_unseen']

        unseen_value_row_index = len(category)
        missing_value_row_index = unseen_value_row_index + 1

        # One more column for missing value when handle is 'column'
        if handle_missing_value == 'column':
            column_count += 1
        if handle_unseen == 'column':
            column_count += 1
        onehotted_cat = np.eye(row_count, column_count, dtype=np.uint8)
        if handle_missing_value == 'ignore':
            onehotted_cat[missing_value_row_index, :] = 0
        if handle_unseen == 'ignore':
            onehotted_cat[unseen_value_row_index, :] = 0
        onehot_index = np.zeros(column.size, dtype=np.uint8)
        missing_value_mask = self._get_missing_value_mask(column)
        one_hotted_cat_index = np.searchsorted(category, column[~missing_value_mask])
        unseen_value_mask = np.take(category, one_hotted_cat_index, mode='clip') != column[~missing_value_mask]
        if np.any(missing_value_mask) and handle_missing_value == 'error':
            raise exceptions.UnexpectedValueError(
                'Encountered missing value {} on index {}'.format(column[missing_value_mask],
                                                                  np.nonzero(column[~missing_value_mask])))
        if np.any(unseen_value_mask) and handle_unseen == 'error':
            raise exceptions.UnexpectedValueError(
                'Encountered unseen value {}'.format(column[~missing_value_mask][unseen_value_mask].values))
        onehot_index[missing_value_mask] = missing_value_row_index
        onehot_index[~missing_value_mask] = one_hotted_cat_index
        onehot_index[~missing_value_mask][unseen_value_mask] = unseen_value_row_index
        one_hot_result = onehotted_cat[onehot_index]
        return one_hot_result

    def _get_missing_value_mask(self, inputs: pd.Series) -> np.array:
        return np.bitwise_or(inputs.isin(self.hyperparams['missing_values']), pd.isnull(inputs))

    def _produce_onehot_column_names(self,
                                     column_index: int,
                                     metadata: metadata_base.DataMetadata, column_name: str) -> typing.Sequence[str]:
        base_column_name = metadata.query_column(column_index).get('name', column_name)
        name_prefix = '{}{}'.format(base_column_name, self.hyperparams['separator'])
        column_names = ['{}{}'.format(name_prefix, cat_name) for cat_name in self._categories[column_index]]
        if self.hyperparams['handle_missing_value'] == 'column':
            column_names.append('{}{}'.format(name_prefix, self._missing_column_name))
        if self.hyperparams['handle_unseen'] == 'column':
            column_names.append('{}{}'.format(name_prefix, self._unseen_column_name))
        return column_names

    def _produce_onehot_metadata(self, inputs: Inputs, outputs: Outputs, column_index: int) -> None:
        for onehot_index in range(outputs.shape[1]):
            outputs.metadata = inputs.metadata.copy_to(
                outputs.metadata,
                (metadata_base.ALL_ELEMENTS, column_index),
                (metadata_base.ALL_ELEMENTS, onehot_index),
            )

        # We set column names based on what Pandas generated.
        for output_column_index, output_column_name in enumerate(outputs.columns):
            outputs.metadata = outputs.metadata.update_column(
                output_column_index,
                {
                    'name': output_column_name,
                },
            )

        # Then we generate the rest of metadata.
        outputs.metadata = outputs.metadata.generate(outputs)

        # Then we unmark output columns as categorical data.
        for output_column_index in range(outputs.shape[1]):
            outputs.metadata = outputs.metadata.remove_semantic_type(
                (metadata_base.ALL_ELEMENTS, output_column_index),
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            )

    def get_params(self) -> Params:
        return Params(
            categories=self._categories,
            fitted=self._fitted
        )

    def set_params(self, *, params: Params) -> None:
        self._categories = params['categories']
        self._fitted = params['fitted']

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_inputs_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            inputs_metadata,
            self.hyperparams['use_columns'],
            self.hyperparams['exclude_columns'],
            can_use_column,
        )
        if not columns_to_use:
            raise ValueError("No column to use.")

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified inputs columns can used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _can_use_inputs_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])

        if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
            # Skip parsing if a column is categorical, but also a target column.
            if not self.hyperparams['encode_target_columns'] and \
                    'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
                return False
            return True
        return False

    def _select_columns(self, inputs: Inputs) -> Tuple[Inputs, List[int]]:
        columns_to_use = self._get_columns(inputs.metadata)

        return inputs.select_columns(columns_to_use), columns_to_use
