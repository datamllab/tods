import copy
import os
import typing

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer


Inputs = container.List
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    match_logic = hyperparams.Enumeration(
        values=['all', 'any'],
        default='any',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should a column have all of semantic types in \"semantic_types\" to be redacted, or any of them?",
    )
    semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Redact columns with these semantic types. Only columns having semantic types listed here will be operated on, based on \"match_logic\".",
    )
    add_semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Semantic types to add to redacted columns. All listed semantic types will be added to all columns which were redacted.",
    )


# TODO: Make clear the assumption that both container type (List) and Datasets should have metadata.
#       Primitive is modifying metadata of Datasets, while there is officially no reason for them
#       to really have metadata: metadata is stored available on the input container type, not
#       values inside it.
class RedactColumnsPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which takes as an input a list of ``Dataset`` objects and redacts values of all columns matching
    a given semantic type or types.

    Redaction is done by setting all values in a redacted column to an empty string.

    It operates only on DataFrame resources inside datasets.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '744c4090-e2f6-489e-8efc-8b1e051bfad6',
            'version': '0.2.0',
            'name': "Redact columns for evaluation",
            'python_path': 'd3m.primitives.tods.evaluation.redact_columns',
            'source': {
                'name': 'DATALab@Texas A&M University',
                'contact': 'mailto:sunmj15@gmail.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/redact_columns.py',
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
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        output_datasets = container.List(generate_metadata=True)

        for dataset in inputs:
            resources = {}
            metadata = dataset.metadata

            for resource_id, resource in dataset.items():
                if not isinstance(resource, container.DataFrame):
                    resources[resource_id] = resource
                    continue

                columns_to_redact = self._get_columns_to_redact(metadata, (resource_id,))

                if not columns_to_redact:
                    resources[resource_id] = resource
                    continue

                resource = copy.copy(resource)

                for column_index in columns_to_redact:
                    column_metadata = dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS, column_index))
                    if 'structural_type' in column_metadata and issubclass(column_metadata['structural_type'], str):
                        resource.iloc[:, column_index] = ''
                    else:
                        raise TypeError("Primitive can operate only on columns with structural type \"str\", not \"{type}\".".format(
                            type=column_metadata.get('structural_type', None),
                        ))

                    metadata = self._update_metadata(metadata, resource_id, column_index, ())

                resources[resource_id] = resource

            dataset = container.Dataset(resources, metadata)

            output_datasets.append(dataset)

        output_datasets.metadata = metadata_base.DataMetadata({
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
            'dimension': {
                'length': len(output_datasets),
            },
        })

        # We update metadata based on metadata of each dataset.
        # TODO: In the future this might be done automatically by generate_metadata.
        #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/119
        for index, dataset in enumerate(output_datasets):
            output_datasets.metadata = dataset.metadata.copy_to(output_datasets.metadata, (), (index,))

        return base.CallResult(output_datasets)

    def _get_columns_to_redact(self, inputs_metadata: metadata_base.DataMetadata, at: metadata_base.Selector) -> typing.Sequence[int]:
        columns = []

        for element in inputs_metadata.get_elements(list(at) + [metadata_base.ALL_ELEMENTS]):
            semantic_types = inputs_metadata.query(list(at) + [metadata_base.ALL_ELEMENTS, element]).get('semantic_types', ())

            # TODO: Should we handle inheritance between semantic types here?
            if self.hyperparams['match_logic'] == 'all':
                matched = all(semantic_type in semantic_types for semantic_type in self.hyperparams['semantic_types'])
            elif self.hyperparams['match_logic'] == 'any':
                matched = any(semantic_type in semantic_types for semantic_type in self.hyperparams['semantic_types'])
            else:
                raise exceptions.UnexpectedValueError("Unknown value of hyper-parameter \"match_logic\": {value}".format(value=self.hyperparams['match_logic']))

            if matched:
                if element is metadata_base.ALL_ELEMENTS:
                    return list(range(inputs_metadata.query(list(at) + [metadata_base.ALL_ELEMENTS]).get('dimension', {}).get('length', 0)))
                else:
                    columns.append(typing.cast(int, element))

        return columns

    def _update_metadata(
        self, inputs_metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment,
        column_index: int, at: metadata_base.Selector,
    ) -> metadata_base.DataMetadata:
        outputs_metadata = inputs_metadata

        for semantic_type in self.hyperparams['add_semantic_types']:
            outputs_metadata = outputs_metadata.add_semantic_type(tuple(at) + (resource_id, metadata_base.ALL_ELEMENTS, column_index), semantic_type)

        return outputs_metadata
