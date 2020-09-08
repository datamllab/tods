import os
from typing import cast, Any, Dict, List, Union, Optional

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, params, hyperparams
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

import pandas  # type: ignore
from pandas.api import types as pandas_types  # type: ignore

import common_primitives

__all__ = ('PandasOneHotEncoderPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    # For each column, a list of category values, sorted.
    categories: Optional[Dict[int, List[Any]]]


class Hyperparams(hyperparams.Hyperparams):
    dummy_na = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Add a column to indicate NaNs, if False NaNs are ignored.",
    )
    drop_first = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether to get k-1 dummies out of k categorical levels by removing the first level.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be encoded, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should encoded columns be appended, should they replace original columns, or should only encoded columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    encode_target_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should it encode also target columns?",
    )


class PandasOneHotEncoderPrimitive(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    One-hot encoder using Pandas implementation.

    """
    __author__ = "Louis Huang"
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f6315ca9-ca39-4e13-91ba-1964ee27281c',
            'version': '0.1.0',
            'name': "Pandas one hot encoder",
            'python_path': 'd3m.primitives.data_preprocessing.one_hot_encoder.PandasCommon',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:luyih@berkeley.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/pandas_onehot_encoder.py',
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
                metadata_base.PrimitiveAlgorithmType.ENCODE_ONE_HOT,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._categories: Dict[int, List[Any]] = {}
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._fitted = False

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])

        if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
            # Skip parsing if a column is categorical, but also a target column.
            if not self.hyperparams['encode_target_columns'] and 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
                return False

            return True

        return False

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)
        # We are OK if no columns ended up being encoded.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can be encoded. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        if self._fitted:
            return CallResult(None)

        columns_to_use = self._get_columns(self._training_inputs.metadata)

        self._categories = {}
        for column_index in columns_to_use:
            self._fit_column(column_index)

        self._fitted = True

        return CallResult(None)

    def _fit_column(self, column_index: int) -> None:
        self._categories[column_index] = sorted(self._training_inputs.iloc[:, column_index].value_counts(dropna=True).index)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        columns_to_use = self._get_columns(inputs.metadata)

        if set(columns_to_use) != set(self._categories.keys()):
            raise exceptions.InvalidArgumentValueError("Columns in provided data do not match fitted columns.")

        outputs_columns = []
        for column_index in columns_to_use:
            outputs_columns.append(self._produce_column(inputs, column_index))

        outputs = base_utils.combine_columns(inputs, columns_to_use, outputs_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return CallResult(outputs)

    def _produce_column(self, inputs: Inputs, column_index: int) -> Outputs:
        # By making a column a category and provide "categories" we can assure same
        # order between multiple calls to "produce".
        input_column = inputs.iloc[:, [column_index]].astype(
            pandas_types.CategoricalDtype(categories=self._categories[column_index]),
        )

        # We first set DataFrame column nam to match one from metadata, if it exists.
        # This then allows Pandas to generate proper new column names.
        input_column.rename({
            input_column.columns[0]: inputs.metadata.query_column(column_index).get('name', input_column.columns[0]),
        }, axis=1, inplace=True)

        output_columns = pandas.get_dummies(
            input_column,
            dummy_na=self.hyperparams['dummy_na'],
            drop_first=self.hyperparams['drop_first'],
        )
        output_columns = container.DataFrame(output_columns, generate_metadata=False)

        # Copy metadata from input column to all output columns.
        for output_column_index in range(len(output_columns.columns)):
            output_columns.metadata = inputs.metadata.copy_to(
                output_columns.metadata,
                (metadata_base.ALL_ELEMENTS, column_index),
                (metadata_base.ALL_ELEMENTS, output_column_index),
            )

        # We set column names based on what Pandas generated.
        for output_column_index, output_column_name in enumerate(output_columns.columns):
            output_columns.metadata = output_columns.metadata.update_column(
                output_column_index,
                {
                    'name': output_column_name,
                },
            )

        # Then we generate the rest of metadata.
        output_columns.metadata = output_columns.metadata.generate(output_columns)

        # Then we unmark output columns as categorical data.
        for output_column_index in range(len(output_columns.columns)):
            output_columns.metadata = output_columns.metadata.remove_semantic_type(
                (metadata_base.ALL_ELEMENTS, output_column_index),
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            )

        return output_columns

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                categories=None,
            )

        return Params(
            categories=self._categories,
        )

    def set_params(self, *, params: Params) -> None:
        self._categories = params['categories']
        self._fitted = all(param is not None for param in params.values())
