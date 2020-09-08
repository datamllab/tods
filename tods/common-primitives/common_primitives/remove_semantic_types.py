import copy
import typing
import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('RemoveSemanticTypesPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices of columns to remove semantic types from.",
    )
    semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Semantic types to remove from columns listed in \"columns\".",
    )


class RemoveSemanticTypesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which removes semantic types from columns in a DataFrame.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '3002bc5b-fa47-4a3d-882e-a8b5f3d756aa',
            'version': '0.1.0',
            'name': "Remove semantic types from columns",
            'python_path': 'd3m.primitives.data_transformation.remove_semantic_types.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/remove_semantic_types.py',
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

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        outputs = copy.copy(inputs)

        outputs.metadata = self._update_metadata(outputs.metadata)

        return base.CallResult(outputs)

    def _update_metadata(self, inputs_metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        outputs_metadata = inputs_metadata

        for column_index in self.hyperparams['columns']:
            for semantic_type in self.hyperparams['semantic_types']:
                outputs_metadata = outputs_metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, column_index), semantic_type)

        return outputs_metadata
