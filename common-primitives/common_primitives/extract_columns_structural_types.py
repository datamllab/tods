import os
import typing

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('ExtractColumnsByStructuralTypesPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    structural_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[typing.Union[str, type]](''),
        default=('str',),
        min_size=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Structural types to use to extract columns. If any of them matches, by default.",
    )
    match_logic = hyperparams.Enumeration(
        values=['all', 'any'],
        default='any',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should a column have all of structural types in \"structural_types\" to be extracted, or any of them?",
    )
    negate = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should columns which do not match structural types in \"structural_types\" be extracted?",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any structural type, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them.",
    )


class ExtractColumnsByStructuralTypesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which extracts columns from input data based on structural types provided.
    Columns which match any of the listed structural types are extracted.

    It uses ``use_columns`` and ``exclude_columns`` to control which columns it considers.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '79674d68-9b93-4359-b385-7b5f60645b06',
            'version': '0.1.0',
            'name': "Extracts columns by structural type",
            'python_path': 'd3m.primitives.data_transformation.extract_columns_by_structural_types.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:ychr93@gmail.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/extract_columns_structural_types.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = inputs.select_columns(columns_to_use)

        outputs = base_utils.combine_columns(inputs, columns_to_use, [output_columns], return_result='new', add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        if 'structural_type' not in column_metadata:
            return False

        structural_types = typing.cast(typing.Sequence[typing.Union[str, type]], self.hyperparams['structural_types'])

        if self.hyperparams['match_logic'] == 'all':
            match = all(d3m_utils.matches_structural_type(column_metadata['structural_type'], structural_type) for structural_type in structural_types)
        elif self.hyperparams['match_logic'] == 'any':
            match = any(d3m_utils.matches_structural_type(column_metadata['structural_type'], structural_type) for structural_type in structural_types)
        else:
            raise exceptions.UnexpectedValueError("Unknown value of hyper-parameter \"match_logic\": {value}".format(value=self.hyperparams['match_logic']))

        if self.hyperparams['negate']:
            return not match
        else:
            return match

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.Sequence[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        if not columns_to_use:
            raise ValueError("Input data has no columns matching structural types: {structural_types}".format(
                structural_types=self.hyperparams['structural_types'],
            ))

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns match structural types. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use
