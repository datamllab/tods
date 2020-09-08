import os
import typing

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('GroupingFieldComposePrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to use when composing a grouping key field.",
    )
    join_char = hyperparams.Hyperparameter[str](
        default="|",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='A string used to join fields.',
    )
    output_name = hyperparams.Hyperparameter[str](
        default="__grouping_key",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The name to use for the new grouping key field.',
    )


class GroupingFieldComposePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which composes suggested grouping key fields into a new single grouping key field.

    The primitve joins the columns marked with SuggestedGroupingKey type in order. The
    resulting value is stored in a new column and marked with the GroupingKey type.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '59db88b9-dd81-4e50-8f43-8f2af959560b',
            'version': '0.1.0',
            'name': "Grouping Field Compose",
            'python_path': 'd3m.primitives.data_transformation.grouping_field_compose.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/grouping_field_compose.py',
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
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def _get_suggested_columns(self, inputs: Inputs) -> typing.Sequence[int]:
        # get every column that has the SuggestedGroupingKey semantic type
        return inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey'])

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        inputs_clone = inputs.copy()
        columns = self.hyperparams['columns']
        output_name = self.hyperparams['output_name']
        join_char = self.hyperparams['join_char']

        # get the columns needing to be joined if not specified by hyperparam
        if len(columns) == 0:
            columns = self._get_suggested_columns(inputs_clone)

        # get the columns needing to be joined if not specified by hyperparam
        if len(columns) == 0:
            self.logger.warning('no columns to use for grouping key so returning input as output')
            return base.CallResult(inputs_clone)

        # join the columns using the separator
        new_col = inputs_clone.iloc[:, list(columns)].apply(lambda x: join_char.join(x), axis=1)

        # append the new colum
        new_col_index = len(inputs_clone.columns)
        inputs_clone.insert(new_col_index, output_name, new_col)

        # update the metadata as needed
        inputs_clone.metadata = inputs_clone.metadata.generate(inputs_clone)
        inputs_clone.metadata = inputs_clone.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, new_col_index), 'http://schema.org/Text')
        inputs_clone.metadata = inputs_clone.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, new_col_index), 'https://metadata.datadrivendiscovery.org/types/GroupingKey')

        return base.CallResult(inputs_clone)
