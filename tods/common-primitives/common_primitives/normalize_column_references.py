import collections
import os

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('NormalizeColumnReferencesPrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset


class Hyperparams(hyperparams.Hyperparams):
    pass


class NormalizeColumnReferencesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts all column references (``foreign_key``, ``boundary_for``, ``confidence_for``)
    found in a dataset to be by column index and not by column name.

    It is useful to do this at the beginning of the pipeline because it is easier to maintain references
    by column index as data and metadata is being changed by the pipeline.

    See for more information `this issue`_.

    .. _this issue: https://gitlab.com/datadrivendiscovery/d3m/issues/343
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '2ee36ea4-5ec3-4c18-909f-9e157fb6d18f',
            'version': '0.1.0',
            'name': "Normalize column references",
            'python_path': 'd3m.primitives.data_transformation.normalize_column_references.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/normalize_column_references.py',
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

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        outputs = inputs.copy()

        for resource_id, resource in inputs.items():
            if not isinstance(resource, container.DataFrame):
                continue

            for column_element in inputs.metadata.get_elements((resource_id, metadata_base.ALL_ELEMENTS,)):
                column_metadata = inputs.metadata.query((resource_id, metadata_base.ALL_ELEMENTS, column_element))

                if 'confidence_for' in column_metadata and 'column_names' in column_metadata['confidence_for']:
                    confidence_for = collections.OrderedDict(column_metadata['confidence_for'])
                    column_reference_resource_id = confidence_for.get('resource_id', resource_id)

                    confidence_for['column_indices'] = [
                        inputs.metadata.get_column_index_from_column_name(column_name, at=(column_reference_resource_id,))
                        for column_name in confidence_for['column_names']
                    ]

                    confidence_for['column_names'] = metadata_base.NO_VALUE

                    outputs.metadata = outputs.metadata.update((resource_id, metadata_base.ALL_ELEMENTS, column_element), {
                        'confidence_for': confidence_for,
                    })

                if 'boundary_for' in column_metadata and 'column_name' in column_metadata['boundary_for']:
                    boundary_for = collections.OrderedDict(column_metadata['boundary_for'])
                    column_reference_resource_id = boundary_for.get('resource_id', resource_id)

                    boundary_for['column_index'] = inputs.metadata.get_column_index_from_column_name(boundary_for['column_name'], at=(column_reference_resource_id,))

                    boundary_for['column_name'] = metadata_base.NO_VALUE

                    outputs.metadata = outputs.metadata.update((resource_id, metadata_base.ALL_ELEMENTS, column_element), {
                        'boundary_for': boundary_for,
                    })

                if 'foreign_key' in column_metadata and column_metadata['foreign_key']['type'] == 'COLUMN' and 'column_name' in column_metadata['foreign_key']:
                    foreign_key = collections.OrderedDict(column_metadata['foreign_key'])
                    column_reference_resource_id = foreign_key['resource_id']

                    foreign_key['column_index'] = inputs.metadata.get_column_index_from_column_name(foreign_key['column_name'], at=(column_reference_resource_id,))

                    foreign_key['column_name'] = metadata_base.NO_VALUE

                    outputs.metadata = outputs.metadata.update((resource_id, metadata_base.ALL_ELEMENTS, column_element), {
                        'foreign_key': foreign_key,
                    })

        return base.CallResult(outputs)
