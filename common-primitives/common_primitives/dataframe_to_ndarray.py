import os
import typing

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('DataFrameToNDArrayPrimitive',)

Inputs = container.DataFrame
Outputs = container.ndarray


class Hyperparams(hyperparams.Hyperparams):
    pass


class DataFrameToNDArrayPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts a pandas dataframe into a numpy array.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '34f71b2e-17bb-488d-a2ba-b60b8c305539',
            'version': '0.1.0',
            'name': "DataFrame to ndarray converter",
            'python_path': 'd3m.primitives.data_transformation.dataframe_to_ndarray.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataframe_to_ndarray.py',
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
        return base.CallResult(container.ndarray(inputs, generate_metadata=True))
