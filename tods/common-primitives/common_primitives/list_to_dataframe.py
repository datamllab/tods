import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('ListToDataFramePrimitive',)

Inputs = container.List
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class ListToDataFramePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts a list into a pandas dataframe.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'dd4598cf-2384-438a-a264-f6c77185132b',
            'version': '0.1.0',
            'name': "List to DataFrame converter",
            'python_path': 'd3m.primitives.data_transformation.list_to_dataframe.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/list_to_dataframe.py',
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
        return base.CallResult(container.DataFrame(inputs, generate_metadata=True))
