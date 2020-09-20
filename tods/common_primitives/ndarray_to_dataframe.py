import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('NDArrayToDataFramePrimitive',)

Inputs = container.ndarray
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class NDArrayToDataFramePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts numpy array into a pandas dataframe.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f5241b2e-64f7-44ad-9675-df3d08066437',
            'version': '0.1.0',
            'name': "ndarray to Dataframe converter",
            'python_path': 'd3m.primitives.data_transformation.ndarray_to_dataframe.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/ndarray_to_dataframe.py',
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
        metadata = inputs.metadata.query((metadata_base.ALL_ELEMENTS,))

        if 'dimension' in metadata:
            # Extract the column names so we can add them to the created dataframe, or set it to index string
            num_cols = inputs.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
            col_names = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, i)).get('name', str(i)) for i in range(num_cols)]
        else:
            col_names = None

        # create a dataframe from the numpy array
        dataframe = container.DataFrame(inputs, columns=col_names, generate_metadata=True)
        return base.CallResult(dataframe)
