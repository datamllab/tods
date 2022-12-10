import os
import typing
import uuid

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.contrib.primitives import compute_scores


__all__ = ('ConstructPredictionsPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If metadata reconstruction happens, this is used for reference columns."
                    " If any specified column is not a primary index or a predicted target, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. If metadata reconstruction happens, this is used for reference columns. Applicable only if \"use_columns\" is not provided.",
    )


class ConstructPredictionsPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which takes as input a DataFrame and outputs a DataFrame in Lincoln Labs predictions
    format: first column is a d3mIndex column (and other primary index columns, e.g., for object detection
    problem), and then predicted targets, each in its column, followed by optional confidence column(s).

    It supports both input columns annotated with semantic types (``https://metadata.datadrivendiscovery.org/types/PrimaryKey``,
    ``https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey``, ``https://metadata.datadrivendiscovery.org/types/PredictedTarget``,
    ``https://metadata.datadrivendiscovery.org/types/Confidence``), or trying to reconstruct metadata.
    This is why the primitive takes also additional input of a reference DataFrame which should
    have metadata to help reconstruct missing metadata. If metadata is missing, the primitive
    assumes that all ``inputs`` columns are predicted targets, without confidence column(s).
    
    Parameters
    ----------
    use_columns: Set
        A set of column indices to force primitive to operate on. If metadata reconstruction happens, this is used for reference columns.
        If any specified column is not a primary index or a predicted target, it is skipped.
    exclude_columns: Set
        A set of column indices to not operate on. If metadata reconstruction happens, this is used for reference columns. Applicable only if \"use_columns\" is not provided.
    """

    
    
    metadata = construct_primitive_metadata(module='data_processing', name='construct_predictions', id='ConstructPredictionsPrimitive', primitive_family='data_transform', description='Construct pipeline predictions output')

    def produce(self, *, inputs: Inputs, reference: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # type: ignore
        index_columns = inputs.metadata.get_index_columns()
        target_columns = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/PredictedTarget',))

        # Target columns cannot be also index columns. This should not really happen,
        # but it could happen with buggy primitives.
        target_columns = [target_column for target_column in target_columns if target_column not in index_columns]

        if index_columns and target_columns:
            outputs = self._produce_using_semantic_types(inputs, index_columns, target_columns)
        else:
            outputs = self._produce_reconstruct(inputs, reference, index_columns, target_columns)

        outputs = compute_scores.ComputeScoresPrimitive._encode_columns(outputs)

        # Generally we do not care about column names in DataFrame itself (but use names of columns from metadata),
        # but in this case setting column names makes it easier to assure that "to_csv" call produces correct output.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/147
        column_names = []
        for column_index in range(len(outputs.columns)):
            column_names.append(outputs.metadata.query_column(column_index).get('name', outputs.columns[column_index]))
        outputs.columns = column_names
        return base.CallResult(outputs)

    def _filter_index_columns(self, inputs_metadata: metadata_base.DataMetadata, index_columns: typing.Sequence[int]) -> typing.Sequence[int]:
        if self.hyperparams['use_columns']: # pragma: no cover
            index_columns = [index_column_index for index_column_index in index_columns if index_column_index in self.hyperparams['use_columns']]
            if not index_columns:
                raise ValueError("No index columns listed in \"use_columns\" hyper-parameter, but index columns are required.")

        else:
            index_columns = [index_column_index for index_column_index in index_columns if index_column_index not in self.hyperparams['exclude_columns']]
            if not index_columns: # pragma: no cover
                raise ValueError("All index columns listed in \"exclude_columns\" hyper-parameter, but index columns are required.")

        names = []
        for index_column in index_columns:
            index_metadata = inputs_metadata.query_column(index_column)
            # We do not care about empty strings for names either.
            if index_metadata.get('name', None):
                names.append(index_metadata['name'])

        if 'd3mIndex' not in names: # pragma: no cover
            raise ValueError("\"d3mIndex\" index column is missing.")

        names_set = set(names)
        if len(names) != len(names_set): # pragma: no cover
            duplicate_names = names
            for name in names_set:
                # Removes just the first occurrence.
                duplicate_names.remove(name)

            self.logger.warning("Duplicate names for index columns: %(duplicate_names)s", {
                'duplicate_names': list(set(duplicate_names)),
            })

        return index_columns

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata, index_columns: typing.Sequence[int], target_columns: typing.Sequence[int]) -> typing.List[int]:
        assert index_columns
        assert target_columns

        index_columns = self._filter_index_columns(inputs_metadata, index_columns)

        if self.hyperparams['use_columns']: # pragma: no cover
            target_columns = [target_column_index for target_column_index in target_columns if target_column_index in self.hyperparams['use_columns']]
            if not target_columns:
                raise ValueError("No target columns listed in \"use_columns\" hyper-parameter, but target columns are required.")

        else:
            target_columns = [target_column_index for target_column_index in target_columns if target_column_index not in self.hyperparams['exclude_columns']]
            if not target_columns: # pragma: no cover
                raise ValueError("All target columns listed in \"exclude_columns\" hyper-parameter, but target columns are required.")

        assert index_columns
        assert target_columns

        return list(index_columns) + list(target_columns)

    def _get_confidence_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        confidence_columns = inputs_metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Confidence',))

        if self.hyperparams['use_columns']:# pragma: no cover
            confidence_columns = [confidence_column_index for confidence_column_index in confidence_columns if confidence_column_index in self.hyperparams['use_columns']]
        else:
            confidence_columns = [confidence_column_index for confidence_column_index in confidence_columns if confidence_column_index not in self.hyperparams['exclude_columns']]

        return confidence_columns

    def _produce_using_semantic_types(self, inputs: Inputs, index_columns: typing.Sequence[int],
                                      target_columns: typing.Sequence[int]) -> Outputs:
        confidence_columns = self._get_confidence_columns(inputs.metadata)

        output_columns = self._get_columns(inputs.metadata, index_columns, target_columns) + confidence_columns

        # "get_index_columns" makes sure that "d3mIndex" is always listed first.
        # And "select_columns" selects columns in order listed, which then
        # always puts "d3mIndex" first.
        outputs = inputs.select_columns(output_columns)

        if confidence_columns:
            outputs.metadata = self._update_confidence_columns(outputs.metadata, confidence_columns)

        return outputs

    def _update_confidence_columns(self, inputs_metadata: metadata_base.DataMetadata, confidence_columns: typing.Sequence[int]) -> metadata_base.DataMetadata: # pragma: no cover
        output_columns_length = inputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        outputs_metadata = inputs_metadata

        # All confidence columns have to be named "confidence".
        for column_index in range(output_columns_length - len(confidence_columns), output_columns_length):
            outputs_metadata = outputs_metadata.update((metadata_base.ALL_ELEMENTS, column_index), {
                'name': 'confidence',
            })

        return outputs_metadata

    def _produce_reconstruct(self, inputs: Inputs, reference: Inputs, index_columns: typing.Sequence[int], target_columns: typing.Sequence[int]) -> Outputs:
        if not index_columns:
            reference_index_columns = reference.metadata.get_index_columns()

            if not reference_index_columns: # pragma: no cover
                raise ValueError("Cannot find an index column in reference data, but index column is required.")

            filtered_index_columns = self._filter_index_columns(reference.metadata, reference_index_columns)
            index = reference.select_columns(filtered_index_columns)
        else: # pragma: no cover
            filtered_index_columns = self._filter_index_columns(inputs.metadata, index_columns)
            index = inputs.select_columns(filtered_index_columns)

        if not target_columns:
            if index_columns: # pragma: no cover
                raise ValueError("No target columns in input data, but index column(s) present.")

            # We assume all inputs are targets.
            targets = inputs

            # We make sure at least basic metadata is generated correctly, so we regenerate metadata.
            targets.metadata = targets.metadata.generate(targets)

            # We set target column names from the reference. We set semantic types.
            targets.metadata = self._update_targets_metadata(targets.metadata, self._get_target_names(reference.metadata))

        else:
            targets = inputs.select_columns(target_columns)

        return index.append_columns(targets)

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, reference: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # pragma: no cover
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, reference=reference)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, reference: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # pragma: no cover
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, reference=reference)

    def _get_target_names(self, metadata: metadata_base.DataMetadata) -> typing.List[typing.Union[str, None]]:
        target_names = []

        for column_index in metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/TrueTarget',)):
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))

            target_names.append(column_metadata.get('name', None))

        return target_names

    def _update_targets_metadata(self, metadata: metadata_base.DataMetadata, target_names: typing.Sequence[typing.Union[str, None]]) -> metadata_base.DataMetadata:
        targets_length = metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        if targets_length != len(target_names): # pragma: no cover
            raise ValueError("Not an expected number of target columns to apply names for. Expected {target_names}, provided {targets_length}.".format(
                target_names=len(target_names),
                targets_length=targets_length,
            ))

        for column_index, target_name in enumerate(target_names):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/Target')
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

            # We do not have it, let's skip it and hope for the best.
            if target_name is None: # pragma: no cover
                continue

            metadata = metadata.update_column(column_index, {
                'name': target_name,
            })

        return metadata
