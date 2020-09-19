import typing
import os

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('ReplaceSemanticTypesPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not have any semantic type from \"from_semantic_types\", it is skipped.",
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
        description="Should columns with replaced semantic types be appended, should they replace original columns, or should only columns with replaced semantic types be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    match_logic = hyperparams.Enumeration(
        values=['all', 'any'],
        default='any',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should a column have all of semantic types in \"from_semantic_types\" to have semantic types replaced, or any of them?",
    )
    from_semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Semantic types to replace. Only columns having semantic types listed here will be operated on, based on \"match_logic\". "
                    "All semantic types listed here will be removed from those columns.",
    )
    to_semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Semantic types to add to matching columns. All listed semantic types will be added to all columns which had semantic types removed.",
    )


class ReplaceSemanticTypesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which replaces semantic types with new semantic types for columns in a DataFrame.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7bae062e-f8b0-4358-91f2-9288a51f3e82',
            'version': '0.2.0',
            'name': "Replace semantic types for columns",
            'python_path': 'd3m.primitives.data_transformation.replace_semantic_types.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/replace_semantic_types.py',
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
        columns_to_use, output_columns = self._produce_columns(inputs)

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])

        if self.hyperparams['match_logic'] == 'all':
            return all(semantic_type in semantic_types for semantic_type in self.hyperparams['from_semantic_types'])
        elif self.hyperparams['match_logic'] == 'any':
            return any(semantic_type in semantic_types for semantic_type in self.hyperparams['from_semantic_types'])
        else:
            raise exceptions.UnexpectedValueError("Unknown value of hyper-parameter \"match_logic\": {value}".format(value=self.hyperparams['match_logic']))

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up having semantic types replaced.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns matches semantic types from \"from_semantic_types\". Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _produce_columns(self, inputs: Inputs) -> typing.Tuple[typing.List[int], typing.List[Outputs]]:
        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = []

        for column_index in columns_to_use:
            column = inputs.select_columns([column_index])
            column.metadata = self._update_metadata(column.metadata)
            output_columns.append(column)

        return columns_to_use, output_columns

    def _produce_columns_metadata(self, inputs_metadata: metadata_base.DataMetadata) -> typing.Tuple[typing.List[int], typing.List[metadata_base.DataMetadata]]:
        columns_to_use = self._get_columns(inputs_metadata)

        output_columns = []

        for column_index in columns_to_use:
            column_metadata = inputs_metadata.select_columns([column_index])
            column_metadata = self._update_metadata(column_metadata)
            output_columns.append(column_metadata)

        return columns_to_use, output_columns

    def _update_metadata(self, inputs_metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        inputs_columns_length = inputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        assert inputs_columns_length == 1, inputs_columns_length

        outputs_metadata = inputs_metadata

        for semantic_type in self.hyperparams['from_semantic_types']:
            outputs_metadata = outputs_metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 0), semantic_type)
        for semantic_type in self.hyperparams['to_semantic_types']:
            outputs_metadata = outputs_metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), semantic_type)

        return outputs_metadata
