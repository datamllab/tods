import os
import typing

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('ListToNDArrayPrimitive',)

Inputs = container.List
Outputs = container.ndarray


class Hyperparams(hyperparams.Hyperparams):
    pass


class ListToNDArrayPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts a list into a numpy array.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '40ff1396-0725-4cf9-b7b9-c6eca6237f65',
            'version': '0.1.0',
            'name': "List to ndarray converter",
            'python_path': 'd3m.primitives.data_transformation.list_to_ndarray.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/list_to_ndarray.py',
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
        dataframe = container.ndarray(inputs, generate_metadata=True)

        # TODO: Remove once fixed in core package and released.
        #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/144
        dataframe.metadata = self._update_metadata(dataframe.metadata)

        return base.CallResult(dataframe)

    def _update_metadata(self, inputs_metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        outputs_metadata = inputs_metadata

        selector: metadata_base.ListSelector = [metadata_base.ALL_ELEMENTS]

        while 'structural_type' in outputs_metadata.query(selector):
            metadata = outputs_metadata.query(selector)
            if issubclass(metadata['structural_type'], (container.List, container.ndarray)):
                outputs_metadata = outputs_metadata.update(selector, {
                    'structural_type': metadata_base.NO_VALUE,
                })
            else:
                break

            selector.append(metadata_base.ALL_ELEMENTS)

        return outputs_metadata.set_table_metadata()
