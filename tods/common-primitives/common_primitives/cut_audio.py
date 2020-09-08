import os
import typing

import frozendict  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column is not an audio column, it is skipped."
                    "Boundary columns are not impacted by this hyper-parameter.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided."
                    "Boundary columns are not impacted by this hyper-parameter.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should columns with cut audio be appended, should they replace original columns, or should only columns with cut audio be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


# TODO: Add a hyper-parameter to remove boundary column(s) when replacing.
class CutAudioPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which uses boundary columns to cut audio columns.

    It uses ``http://schema.org/AudioObject`` and structural type ``container.ndarray` to
    find columns with audio data.

    It searches for boundary columns referencing them.
    Boundary columns are identified by ``https://metadata.datadrivendiscovery.org/types/Interval``,
    ``https://metadata.datadrivendiscovery.org/types/IntervalStart`` and
    ``https://metadata.datadrivendiscovery.org/types/IntervalEnd`` semantic types.

    It requires that the audio dimension has ``sampling_rate`` metadata set.

    Boundaries are rounded down to samples. Cut is done exclusive: not including the last sample.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '4ad9ce62-283d-4765-a87b-78b55d89a4ed',
            'version': '0.1.0',
            'name': 'Cut audio columns',
            'python_path': 'd3m.primitives.data_transformation.cut_audio.Common',
            'keywords': ['audio', 'cut'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/cut_audio.py',
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
                metadata_base.PrimitiveAlgorithmType.AUDIO_STREAM_MANIPULATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        audio_columns_to_use = self._get_audio_columns(inputs.metadata)

        all_boundary_columns = self._get_boundary_columns(inputs.metadata, audio_columns_to_use)

        output_columns = [self._produce_column(inputs, audio_column, boundary_columns) for audio_column, boundary_columns in all_boundary_columns.items()]

        outputs = base_utils.combine_columns(
            inputs, list(all_boundary_columns.keys()), output_columns,
            return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'],
        )

        if self.hyperparams['return_result'] == 'replace':
            outputs.metadata = self._remove_metadata_references(outputs.metadata, all_boundary_columns)

        return base.CallResult(outputs)

    def _remove_metadata_references(self, inputs_metadata: metadata_base.DataMetadata, all_boundary_columns: typing.Dict[int, typing.List[int]]) -> metadata_base.DataMetadata:
        outputs_metadata = inputs_metadata

        # When replacing, boundary columns do not apply anymore to new columns.
        for audio_column, boundary_columns in all_boundary_columns.items():
            for boundary_column in boundary_columns:
                outputs_metadata = outputs_metadata.update_column(boundary_column, {
                    'boundary_for': metadata_base.NO_VALUE,
                })

        return outputs_metadata

    def _produce_column(self, inputs: Inputs, column_index: int, boundary_columns: typing.List[int]) -> Outputs:
        cut_audio = []
        for row_index, value in enumerate(inputs.iloc[:, column_index]):
            try:
                if len(boundary_columns) == 1:
                    # Float vector is a ndarray vector, so we convert it to a list.
                    boundaries = list(inputs.iloc[row_index, boundary_columns[0]])
                else:
                    assert len(boundary_columns) == 2

                    boundaries = [inputs.iloc[row_index, boundary_columns[0]], inputs.iloc[row_index, boundary_columns[1]]]

                cut_audio.append(self._cut_audio(boundaries, inputs.metadata.query((row_index, column_index)), value))

            except Exception as error:
                raise ValueError("Could not cut audio in column {column_index} at row {row_index}.".format(
                    column_index=column_index,
                    row_index=row_index,
                )) from error

        column = container.DataFrame({inputs.columns[column_index]: cut_audio}, generate_metadata=False)

        column.metadata = self._produce_column_metadata(inputs.metadata, column_index, cut_audio)
        column.metadata = column.metadata.generate(column)

        return column

    def _produce_column_metadata(self, inputs_metadata: metadata_base.DataMetadata, column_index: int,
                                 cut_audio: typing.Sequence[container.ndarray]) -> metadata_base.DataMetadata:
        column_metadata = inputs_metadata.select_columns([column_index])

        for row_index, audio in enumerate(cut_audio):
            column_metadata = column_metadata.update((row_index, 0), {
                'dimension': {
                    'length': len(audio),
                }
            })

        return column_metadata

    def _cut_audio(self, boundaries: typing.List[int], metadata: frozendict.FrozenOrderedDict, audio: container.ndarray) -> container.ndarray:
        if 'sampling_rate' not in metadata.get('dimension', {}):
            raise ValueError("\"sampling_rate\" dimension metadata is missing.")

        sampling_rate = metadata['dimension']['sampling_rate']

        assert len(boundaries) == 2

        start = int(sampling_rate * boundaries[0])
        end = int(sampling_rate * boundaries[1])

        if not 0 <= start <= end:
            self.logger.warning("Interval start is out of range: start=%(start)s, end=%(end)s, length=%(length)s", {
                'start': start,
                'end': end,
                'length': len(audio),
            })
        if not start <= end <= len(audio):
            self.logger.warning("Interval end is out of range: start=%(start)s, end=%(end)s, length=%(length)s", {
                'start': start,
                'end': end,
                'length': len(audio),
            })

        return audio[start:end]

    def _can_use_audio_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        if not issubclass(column_metadata['structural_type'], container.ndarray):
            return False

        if 'http://schema.org/AudioObject' not in column_metadata.get('semantic_types', []):
            return False

        return True

    def _get_audio_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_audio_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being cut.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns contain audio. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _get_boundary_columns(self, inputs_metadata: metadata_base.DataMetadata, audio_columns: typing.List[int]) -> typing.Dict[int, typing.List[int]]:
        # In Python 3.6 this dict has deterministic order.
        boundary_columns = {}
        for audio_column in audio_columns:
            boundary_columns_for_column = self._get_boundary_columns_for_column(inputs_metadata, audio_column)

            if boundary_columns_for_column:
                boundary_columns[audio_column] = boundary_columns_for_column
            else:
                # This is OK, not all audio columns should be cut.
                self.logger.debug("Audio column %(audio_column)s does not have boundary columns.", {
                    'audio_column': audio_column,
                })

        return boundary_columns

    def _get_boundary_columns_for_column(self, inputs_metadata: metadata_base.DataMetadata, audio_column: int) -> typing.List[int]:
        """
        If returned list contains one element, then that column is "interval" column.
        If it contains two elements, then the first column is "interval start" column, and the second
        "interval end" column.
        """

        columns_length = inputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        # In Python 3.6 this dict has deterministic order.
        boundary_columns_with_index = {}

        for column_index in range(columns_length):
            column_metadata = inputs_metadata.query_column(column_index)
            semantic_types = column_metadata.get('semantic_types', [])

            if not any(semantic_type in semantic_types for semantic_type in [
                'https://metadata.datadrivendiscovery.org/types/Interval',
                'https://metadata.datadrivendiscovery.org/types/IntervalStart',
                'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
            ]):
                continue

            if audio_column == column_metadata.get('boundary_for', {}).get('column_index', None):
                boundary_columns_with_index[column_index] = column_metadata

        if not boundary_columns_with_index:
            return []

        if len(boundary_columns_with_index) == 1:
            for column_index, column_metadata in boundary_columns_with_index.items():
                semantic_types = column_metadata.get('semantic_types', [])

                if any(semantic_type in semantic_types for semantic_type in [
                    'https://metadata.datadrivendiscovery.org/types/IntervalStart',
                    'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
                ]):
                    self.logger.warning("One boundary column %(boundary_column)s for audio column %(audio_column)s, but invalid semantic types.", {
                        'boundary_column': column_index,
                        'audio_column': audio_column,
                    })
                    return []

                assert 'https://metadata.datadrivendiscovery.org/types/Interval' in semantic_types, column_index

                return [column_index]

        elif len(boundary_columns_with_index) == 2:
            start_column_index: int = None
            end_column_index: int = None

            for column_index, column_metadata in boundary_columns_with_index.items():
                semantic_types = column_metadata.get('semantic_types', [])

                if 'https://metadata.datadrivendiscovery.org/types/Interval' in semantic_types:
                    self.logger.warning("Two boundary columns %(boundary_columns)s for audio column %(audio_column)s, but boundary column %(boundary_column)s has invalid semantic type.", {
                        'boundary_columns': list(boundary_columns_with_index.keys()),
                        'boundary_column': column_index,
                        'audio_column': audio_column,
                    })
                    return []

                # It is OK if set one of the variables twice, then the other one will stay "None"
                # and we will abort below.
                if 'https://metadata.datadrivendiscovery.org/types/IntervalStart' in semantic_types:
                    start_column_index = column_index
                elif 'https://metadata.datadrivendiscovery.org/types/IntervalEnd' in semantic_types:
                    end_column_index = column_index
                else:
                    assert False, column_index

            if start_column_index is not None and end_column_index is not None:
                return [start_column_index, end_column_index]
            else:
                self.logger.warning("Two boundary columns %(boundary_columns)s for audio column %(audio_column)s, but invalid semantic types.", {
                    'boundary_columns': list(boundary_columns_with_index.keys()),
                    'audio_column': audio_column,
                })
                return []

        else:
            self.logger.warning("Multiple (%(count)s) boundary columns for audio column %(audio_column)s.".format({
                'count': len(boundary_columns_with_index),
                'audio_column': audio_column,
            }))
            return []

        # Not really necessary, but mypy is happier with it.
        return []
