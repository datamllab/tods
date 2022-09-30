import hashlib
import os
import typing
import uuid

import numpy  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

from tods.data_processing import utils

__all__ = ('ColumnParserPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    parse_semantic_types = hyperparams.Set(
        elements=hyperparams.Enumeration(
            values=[
                'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
            ],
            # Default is ignored.
            # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
            default='http://schema.org/Boolean',
        ),
        default=(
            'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            'http://schema.org/Integer', 'http://schema.org/Float',
            'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
        ),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of semantic types to parse. One can provide a subset of supported semantic types to limit what the primitive parses.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
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
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    parse_categorical_target_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should it parse also categorical target columns?",
    )
    replace_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Replace primary index columns even if otherwise appending columns. Applicable only if \"return_result\" is set to \"append\".",
    )
    fuzzy_time_parsing = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Use fuzzy time parsing.",
    )


class ColumnParserPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which parses strings into their parsed values.

    It goes over all columns (by default, controlled by ``use_columns``, ``exclude_columns``)
    and checks those with structural type ``str`` if they have a semantic type suggesting
    that they are a boolean value, categorical, integer, float, or time (by default,
    controlled by ``parse_semantic_types``). Categorical values are converted with
    hash encoding.

    What is returned is controlled by ``return_result`` and ``add_index_columns``.

    Parameters
    ----------
    parse_semantic_types :Set
        A set of semantic types to parse. One can provide a subset of supported semantic types to limit what the primitive parses.       
    use_columns :Set
        A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.
    exclude_columns :Set
        A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.
    return_result :Enumeration
        Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned?
    add_index_columns :Bool
        Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".
    parse_categorical_target_columns = hyperparams.UniformBool
        Should it parse also categorical target columns?
    replace_index_columns :Bool
        Replace primary index columns even if otherwise appending columns. Applicable only if \"return_result\" is set to \"append\".   
    fuzzy_time_parsing :Bool
        Use fuzzy time parsing.
    """

   
    metadata = construct_primitive_metadata(module='data_processing', name='column_parser', id='ColumnParserPrimitive', primitive_family='data_transform', description='Parses strings into their types')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        columns_to_use, output_columns = self._produce_columns(inputs)

        if self.hyperparams['replace_index_columns'] and self.hyperparams['return_result'] == 'append': # pragma: no cover
            assert len(columns_to_use) == len(output_columns)

            index_columns = inputs.metadata.get_index_columns()

            index_columns_to_use = []
            other_columns_to_use = []
            index_output_columns = []
            other_output_columns = []
            for column_to_use, output_column in zip(columns_to_use, output_columns):
                if column_to_use in index_columns:
                    index_columns_to_use.append(column_to_use)
                    index_output_columns.append(output_column)
                else:
                    other_columns_to_use.append(column_to_use)
                    other_output_columns.append(output_column)

            outputs = base_utils.combine_columns(inputs, index_columns_to_use, index_output_columns, return_result='replace', add_index_columns=self.hyperparams['add_index_columns'])
            outputs = base_utils.combine_columns(outputs, other_columns_to_use, other_output_columns, return_result='append', add_index_columns=self.hyperparams['add_index_columns'])
        else:
            outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:   # pragma: no cover
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        # We produce only on columns which have not yet been parsed (are strings).
        if column_metadata['structural_type'] != str:
            return False

        semantic_types = column_metadata.get('semantic_types', [])

        for semantic_type in self.hyperparams['parse_semantic_types']:
            if semantic_type not in semantic_types:
                continue

            if semantic_type == 'https://metadata.datadrivendiscovery.org/types/CategoricalData':
                # Skip parsing if a column is categorical, but also a target column.
                if not self.hyperparams['parse_categorical_target_columns'] and 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
                    continue

            return True

        return False

    def _produce_columns(self, inputs: Inputs) -> typing.Tuple[typing.List[int], typing.List[Outputs]]:   # pragma: no cover
        # The logic of parsing values tries to mirror also the logic of detecting
        # values in "SimpleProfilerPrimitive". One should keep them in sync.

        columns_to_use = self._get_columns(inputs.metadata)

        # We check against this list again, because there might be multiple matching semantic types
        # (which is not really valid).
        parse_semantic_types = self.hyperparams['parse_semantic_types']

        output_columns = []

        for column_index in columns_to_use:
            column_metadata = inputs.metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = column_metadata.get('semantic_types', [])
            if column_metadata['structural_type'] == str:
                if 'http://schema.org/Boolean' in parse_semantic_types and 'http://schema.org/Boolean' in semantic_types:
                    output_columns.append(self._parse_boolean_data(inputs, column_index))

                elif 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in parse_semantic_types and \
                        'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types and \
                        (self.hyperparams['parse_categorical_target_columns'] or 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types):
                    output_columns.append(self._parse_categorical_data(inputs, column_index))

                elif 'http://schema.org/Integer' in parse_semantic_types and 'http://schema.org/Integer' in semantic_types:
                    # For primary key we know all values have to exist so we can assume they can always be represented as integers.
                    if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in semantic_types:
                        integer_required = True
                    else:
                        integer_required = False

                    output_columns.append(self._parse_integer(inputs, column_index, integer_required))

                elif 'http://schema.org/Float' in parse_semantic_types and 'http://schema.org/Float' in semantic_types:
                    output_columns.append(self._parse_float_data(inputs, column_index))

                elif 'https://metadata.datadrivendiscovery.org/types/FloatVector' in parse_semantic_types and 'https://metadata.datadrivendiscovery.org/types/FloatVector' in semantic_types:
                    output_columns.append(self._parse_float_vector_data(inputs, column_index))

                elif 'http://schema.org/DateTime' in parse_semantic_types and 'http://schema.org/DateTime' in semantic_types:
                    output_columns.append(self._parse_time_data(inputs, column_index, self.hyperparams['fuzzy_time_parsing']))

                else:
                    assert False, column_index

        assert len(output_columns) == len(columns_to_use)

        return columns_to_use, output_columns

    def _produce_columns_metadata(self, inputs_metadata: metadata_base.DataMetadata) -> typing.Tuple[typing.List[int], typing.List[metadata_base.DataMetadata]]: # pragma: no cover
        columns_to_use = self._get_columns(inputs_metadata)

        # We check against this list again, because there might be multiple matching semantic types
        # (which is not really valid).
        parse_semantic_types = self.hyperparams['parse_semantic_types']

        output_columns = []

        for column_index in columns_to_use:
            column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = column_metadata.get('semantic_types', [])
            if column_metadata['structural_type'] == str:
                if 'http://schema.org/Boolean' in parse_semantic_types and 'http://schema.org/Boolean' in semantic_types:
                    output_columns.append(self._parse_boolean_metadata(inputs_metadata, column_index))

                elif 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in parse_semantic_types and \
                        'https://metadata.datadrivendiscovery.org/types/CategoricalData' in semantic_types and \
                        (self.hyperparams['parse_categorical_target_columns'] or 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types):
                    output_columns.append(self._parse_categorical_metadata(inputs_metadata, column_index))

                elif 'http://schema.org/Integer' in parse_semantic_types and 'http://schema.org/Integer' in semantic_types:
                    output_columns.append(self._parse_integer_metadata(inputs_metadata, column_index))

                elif 'http://schema.org/Float' in parse_semantic_types and 'http://schema.org/Float' in semantic_types:
                    output_columns.append(self._parse_float_metadata(inputs_metadata, column_index))

                elif 'https://metadata.datadrivendiscovery.org/types/FloatVector' in parse_semantic_types and 'https://metadata.datadrivendiscovery.org/types/FloatVector' in semantic_types: # pragma: no cover
                    output_columns.append(self._parse_float_vector_metadata(inputs_metadata, column_index))

                elif 'http://schema.org/DateTime' in parse_semantic_types and 'http://schema.org/DateTime' in semantic_types: # pragma: no cover
                    output_columns.append(self._parse_time_metadata(inputs_metadata, column_index))

                else:
                    assert False, column_index

        assert len(output_columns) == len(columns_to_use)

        return columns_to_use, output_columns

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being parsed.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can parsed. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    @classmethod
    def _parse_boolean_data(cls, inputs: Inputs, column_index: metadata_base.SimpleSelectorSegment) -> Outputs:
        return cls._parse_categorical_data(inputs, column_index) # pragma: no cover

    @classmethod
    def _parse_boolean_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_index: metadata_base.SimpleSelectorSegment) -> metadata_base.DataMetadata:
        return cls._parse_categorical_metadata(inputs_metadata, column_index) # pragma: no cover

    @classmethod
    def _parse_categorical_data(cls, inputs: Inputs, column_index: metadata_base.SimpleSelectorSegment) -> Outputs: # pragma: no cover
        values_map: typing.Dict[str, int] = {}
        for value in inputs.iloc[:, column_index]:
            value = value.strip()
            if value not in values_map:
                value_hash = hashlib.sha256(value.encode('utf8'))
                values_map[value] = int.from_bytes(value_hash.digest()[0:8], byteorder='little') ^ int.from_bytes(value_hash.digest()[8:16], byteorder='little') ^ \
                    int.from_bytes(value_hash.digest()[16:24], byteorder='little') ^ int.from_bytes(value_hash.digest()[24:32], byteorder='little')

        outputs = container.DataFrame({inputs.columns[column_index]: [values_map[value.strip()] for value in inputs.iloc[:, column_index]]}, generate_metadata=False)
        outputs.metadata = cls._parse_categorical_metadata(inputs.metadata, column_index)

        return outputs

    @classmethod
    def _parse_categorical_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_index: metadata_base.SimpleSelectorSegment) -> metadata_base.DataMetadata: # pragma: no cover
        outputs_metadata = inputs_metadata.select_columns([column_index])
        return outputs_metadata.update_column(0, {'structural_type': int})

    @classmethod
    def _str_to_int(cls, value: str) -> typing.Union[float, int]:
        try:
            return int(value.strip())
        except ValueError:
            try:
                # Maybe it is an int represented as a float. Let's try this. This can get rid of non-integer
                # part of the value, but the integer was requested through a semantic type, so this is probably OK.
                return int(float(value.strip()))
            except ValueError:
                # No luck, use NaN to represent a missing value.
                return float('nan')

    @classmethod
    def _parse_integer(cls, inputs: Inputs, column_index: metadata_base.SimpleSelectorSegment,
                       integer_required: bool) -> container.DataFrame:
        outputs = container.DataFrame({inputs.columns[column_index]: [cls._str_to_int(value) for value in inputs.iloc[:, column_index]]}, generate_metadata=False)

        if outputs.dtypes.iloc[0].kind == 'f':
            structural_type: type = float
        elif outputs.dtypes.iloc[0].kind in ['i', 'u']:
            structural_type = int
        else:
            assert False, outputs.dtypes.iloc[0]

        if structural_type is float and integer_required:
            raise ValueError("Not all values in a column can be parsed into integers, but only integers were expected.")

        outputs.metadata = inputs.metadata.select_columns([column_index])
        outputs.metadata = outputs.metadata.update_column(0, {'structural_type': structural_type})

        return outputs

    @classmethod
    def _parse_integer_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_index: metadata_base.SimpleSelectorSegment) -> metadata_base.DataMetadata: # pragma: no cover
        outputs_metadata = inputs_metadata.select_columns([column_index])
        # Without data we assume we can parse everything into integers. This might not be true and
        # we might end up parsing into floats if we have to represent missing (or invalid) values.
        return outputs_metadata.update_column(0, {'structural_type': int})

    @classmethod
    def _str_to_float(cls, value: str) -> float:
        try:
            return float(value.strip())
        except ValueError: # pragma: no cover
            return float('nan')

    @classmethod
    def _parse_float_data(cls, inputs: Inputs, column_index: metadata_base.SimpleSelectorSegment) -> Outputs:
        outputs = container.DataFrame({inputs.columns[column_index]: [cls._str_to_float(value) for value in inputs.iloc[:, column_index]]}, generate_metadata=False)
        outputs.metadata = cls._parse_float_metadata(inputs.metadata, column_index)

        return outputs

    @classmethod
    def _parse_float_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_index: metadata_base.SimpleSelectorSegment) -> metadata_base.DataMetadata:
        outputs_metadata = inputs_metadata.select_columns([column_index])
        return outputs_metadata.update_column(0, {'structural_type': float})

    @classmethod
    def _parse_float_vector_data(cls, inputs: Inputs, column_index: metadata_base.SimpleSelectorSegment) -> Outputs: # pragma: no cover
        # We are pretty strict here because we are assuming this was generated programmatically.
        outputs = container.DataFrame(
            {
                inputs.columns[column_index]: [
                    container.ndarray([cls._str_to_float(value) for value in values.split(',')])
                    for values in inputs.iloc[:, column_index]
                ],
            },
            generate_metadata=False,
        )
        outputs.metadata = cls._parse_float_metadata(inputs.metadata, column_index)
        # We have to automatically generate metadata to set ndarray dimension(s).
        outputs.metadata = outputs.metadata.generate(outputs)

        return outputs

    @classmethod
    def _parse_float_vector_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_index: metadata_base.SimpleSelectorSegment) -> metadata_base.DataMetadata: # pragma: no cover
        outputs_metadata = inputs_metadata.select_columns([column_index])
        # We cannot know the dimension of the ndarray without data.
        outputs_metadata = outputs_metadata.update_column(0, {'structural_type': container.ndarray})
        outputs_metadata = outputs_metadata.update((metadata_base.ALL_ELEMENTS, 0, metadata_base.ALL_ELEMENTS), {'structural_type': numpy.float64})
        return outputs_metadata

    @classmethod
    def _parse_time_data(cls, inputs: Inputs, column_index: metadata_base.SimpleSelectorSegment, fuzzy: bool) -> Outputs: # pragma: no cover
        outputs = container.DataFrame({inputs.columns[column_index]: [utils.parse_datetime_to_float(value, fuzzy=fuzzy) for value in inputs.iloc[:, column_index]]}, generate_metadata=False)
        outputs.metadata = cls._parse_time_metadata(inputs.metadata, column_index)

        return outputs

    @classmethod
    def _parse_time_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_index: metadata_base.SimpleSelectorSegment) -> metadata_base.DataMetadata: # pragma: no cover
        outputs_metadata = inputs_metadata.select_columns([column_index])
        return outputs_metadata.update_column(0, {'structural_type': float})
