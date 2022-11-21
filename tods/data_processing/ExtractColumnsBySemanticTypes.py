import os
import typing
import uuid

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

__all__ = ('ExtractColumnsBySemanticTypesPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=('https://metadata.datadrivendiscovery.org/types/Attribute',),
        min_size=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Semantic types to use to extract columns. If any of them matches, by default.",
    )
    match_logic = hyperparams.Enumeration(
        values=['all', 'any', 'equal'],
        default='any',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should a column have all of semantic types in \"semantic_types\" to be extracted, or any of them?",
    )
    negate = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should columns which do not match semantic types in \"semantic_types\" be extracted?",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
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


class ExtractColumnsBySemanticTypesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which extracts columns from input data based on semantic types provided.
    Columns which match any of the listed semantic types are extracted.

    If you want to extract only attributes, you can use ``https://metadata.datadrivendiscovery.org/types/Attribute``
    semantic type (also default).

    For real targets (not suggested targets) use ``https://metadata.datadrivendiscovery.org/types/Target``.
    For this to work, columns have to be are marked as targets by the TA2 in a dataset before passing the dataset
    through a pipeline. Or something else has to mark them at some point in a pipeline.

    It uses ``use_columns`` and ``exclude_columns`` to control which columns it considers.
    
    Parameters
    -----------
    semantic_types :Set
        Semantic types to use to extract columns. If any of them matches, by default.
    match_logic :Enumeration
        Should a column have all of semantic types in \"semantic_types\" to be extracted, or any of them?
    negate : Bool
        Should columns which do not match semantic types in \"semantic_types\" be extracted?    
    use_columns : Set
        A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.
    exclude_columns :Set
        A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.
    add_index_columns :Bool
        Also include primary index columns if input data has them.
    """

    
    metadata = construct_primitive_metadata(module='data_processing', name='extract_columns_by_semantic_types', id='ExtractColumnsBySemanticTypesPrimitive', primitive_family='data_transform', description='Extracts columns by semantic type')

    
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = inputs.select_columns(columns_to_use)

        outputs = base_utils.combine_columns(inputs, columns_to_use, [output_columns], return_result='new', add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        semantic_types = column_metadata.get('semantic_types', [])

        if self.hyperparams['match_logic'] == 'all': # pragma: no cover
            match = all(semantic_type in semantic_types for semantic_type in self.hyperparams['semantic_types'])
        elif self.hyperparams['match_logic'] == 'any':
            match = any(semantic_type in semantic_types for semantic_type in self.hyperparams['semantic_types'])
        elif self.hyperparams["match_logic"] == "equal": # pragma: no cover
            match = set(semantic_types) == set(self.hyperparams["semantic_types"])
        else:
            raise exceptions.UnexpectedValueError("Unknown value of hyper-parameter \"match_logic\": {value}".format(value=self.hyperparams['match_logic']))

        if self.hyperparams['negate']: # pragma: no cover
            return not match
        else:
            return match

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.Sequence[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        if not columns_to_use:
            raise ValueError("Input data has no columns matching semantic types: {semantic_types}".format(
                semantic_types=self.hyperparams['semantic_types'],
            ))

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns match semantic types. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use
