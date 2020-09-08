import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('NDArrayToListPrimitive',)

Inputs = container.ndarray
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    pass


class NDArrayToListPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts a numpy array into a list.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'b5101331-64b4-451a-beb6-260b40d1436b',
            'version': '0.1.0',
            'name': "ndarray to list converter",
            'python_path': 'd3m.primitives.data_transformation.ndarray_to_list.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/ndarray_to_list.py',
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
        return base.CallResult(container.List(inputs, generate_metadata=True))
