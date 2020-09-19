import os
from typing import cast, Any, Dict, List, Union, Optional

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, params, hyperparams
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

import common_primitives

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    # For each column, a map between original labels and encoded values.
    labels: Optional[Dict[int, Dict[str, int]]]
    # For each column, a map between encoded values and original labels.
    inverse_labels: Optional[Dict[int, Dict[int, str]]]


class Hyperparams(hyperparams.Hyperparams):
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


class UnseenLabelEncoderPrimitive(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Label encoder that can puts any unseen categories into a single category.
    """

    __author__ = "Brown"
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '26fc8fd3-f6b2-4c65-8afb-edb54ed2a3e4',
            'version': '0.2.0',
            'name': "Label encoder with an unseen category",
            'python_path': 'd3m.primitives.data_preprocessing.label_encoder.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:yeounoh_chung@brown.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/unseen_label_encoder.py',
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
                metadata_base.PrimitiveAlgorithmType.CATEGORY_ENCODER,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._labels: Dict[int, Dict[Any, int]] = {}
        self._inverse_labels: Dict[int, Dict[int, Any]] = {}
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._fitted = False

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        # We produce only on columns which have not yet been encoded (are strings).
        if column_metadata['structural_type'] != str:
            return False

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

        self._labels = {}
        self._inverse_labels = {}

        for column_index in columns_to_use:
            self._fit_column(column_index)

        self._fitted = True

        return CallResult(None)

    def _fit_column(self, column_index: int) -> None:
        self._labels[column_index] = {}
        self._inverse_labels[column_index] = {}

        for value in self._training_inputs.iloc[:, column_index]:
            value = value.strip()
            if value not in self._labels[column_index]:
                # We add 1 to reserve 0.
                new_label = len(self._labels[column_index]) + 1
                self._labels[column_index][value] = new_label
                self._inverse_labels[column_index][new_label] = value

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = [self._produce_column(inputs, column_index) for column_index in columns_to_use]

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return CallResult(outputs)

    def _produce_column(self, inputs: Inputs, column_index: int) -> Outputs:
        column = container.DataFrame([self._labels[column_index].get(value.strip(), 0) for value in inputs.iloc[:, column_index]], generate_metadata=False)

        column.metadata = self._produce_column_metadata(inputs.metadata, column_index)

        return column

    def _produce_column_metadata(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> metadata_base.DataMetadata:
        column_metadata = inputs_metadata.select_columns([column_index])
        column_metadata = column_metadata.update_column(0, {'structural_type': int})

        return column_metadata

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                labels=None,
                inverse_labels=None,
            )

        return Params(
            labels=self._labels,
            inverse_labels=self._inverse_labels,
        )

    def set_params(self, *, params: Params) -> None:
        self._labels = params['labels']
        self._inverse_labels = params['inverse_labels']
        self._fitted = all(param is not None for param in params.values())
