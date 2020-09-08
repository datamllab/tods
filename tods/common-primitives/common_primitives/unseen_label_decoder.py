import os
from typing import cast, Dict, List, Union, Optional

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

import common_primitives
from common_primitives import unseen_label_encoder

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    encoder = hyperparams.Primitive(
        default=unseen_label_encoder.UnseenLabelEncoderPrimitive,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="An \"UnseenLabelEncoderPrimitive\" to use for decoding.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be decoded, it is skipped.",
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
        description="Should decoded columns be appended, should they replace original columns, or should only decoded columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


# TODO: This is not yet very useful because it currently requires that columns are at the same index when decoding.
#       This should be done better once each column has an unique ID.
#       Then we can store mapping using that ID instead of column index.
#       Alternatively, inverse mapping could be stored into metadata.
#       See: https://gitlab.com/datadrivendiscovery/d3m/issues/112
class UnseenLabelDecoderPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which inverses the label encoding by ``UnseenLabelEncoderPrimitive``.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '39ae30f7-39ed-40af-8679-5cf108499605',
            'version': '0.1.0',
            'name': "Label decoder for UnseenLabelEncoderPrimitive",
            'python_path': 'd3m.primitives.data_preprocessing.label_decoder.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/unseen_label_decoder.py',
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

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        # We produce only on columns which have been encoded (are integers).
        if column_metadata['structural_type'] != int:
            return False

        semantic_types = column_metadata.get('semantic_types', [])

        if 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types:
            return True

        return False

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being decoded.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can be decoded. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = [self._produce_column(inputs, column_index) for column_index in columns_to_use]

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return CallResult(outputs)

    def _produce_column(self, inputs: Inputs, column_index: int) -> Outputs:
        inverse_labels = self.hyperparams['encoder'].get_params()['inverse_labels']

        # We use an empty string for all labels we cannot decode.
        column = container.DataFrame([inverse_labels[column_index].get(value, '') for value in inputs.iloc[:, column_index]], generate_metadata=False)

        column.metadata = self._produce_column_metadata(inputs.metadata, column_index)

        return column

    def _produce_column_metadata(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> metadata_base.DataMetadata:
        column_metadata = inputs_metadata.select_columns([column_index])
        column_metadata = column_metadata.update_column(0, {'structural_type': str})

        return column_metadata
