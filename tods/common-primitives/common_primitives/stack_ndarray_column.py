import copy
import os
import typing

import numpy  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('StackNDArrayColumnPrimitive',)

Inputs = container.DataFrame
Outputs = container.ndarray


class Hyperparams(hyperparams.Hyperparams):
    use_column = hyperparams.Hyperparameter[typing.Optional[int]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A column index to force primitive to operate on. If the specified column is not a column of numpy arrays, an error is raised.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )


class StackNDArrayColumnPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which stacks numpy arrays in a column and returns a stacked numpy array along the new 0 axis.

    All arrays must have the same shape.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '48c99864-14f3-4a61-b3a6-e439f22825f6',
            'version': '0.1.0',
            'name': "Stack numpy arrays in a column",
            'python_path': 'd3m.primitives.data_transformation.stack_ndarray_column.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/stack_ndarray_column.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        column_to_use = self._get_column(inputs.metadata)

        outputs = container.ndarray(numpy.stack(inputs.iloc[:, column_to_use], axis=0), generate_metadata=False)

        outputs.metadata = self._update_metadata(inputs.metadata, column_to_use)

        # Update the structure.
        outputs.metadata = outputs.metadata.generate(outputs)

        return base.CallResult(outputs)

    def _update_metadata(self, inputs_metadata: metadata_base.DataMetadata, column_to_use: int) -> metadata_base.DataMetadata:
        # Copy input metadata so that we can modify it in-place.
        outputs_metadata = copy.copy(inputs_metadata)
        outputs_metadata._current_metadata = inputs_metadata._current_metadata.copy()

        # Remove columns dimension and replace it with metadata of the column.
        # TODO: Move this to metadata API.
        all_columns_metadata_entry = outputs_metadata._current_metadata.all_elements.all_elements
        column_metadata_entry = outputs_metadata._current_metadata.all_elements.elements[column_to_use]
        if all_columns_metadata_entry is not None:
            outputs_metadata._current_metadata.all_elements = outputs_metadata._merge_metadata_entries(all_columns_metadata_entry, column_metadata_entry)
        else:
            outputs_metadata._current_metadata.all_elements = column_metadata_entry
        outputs_metadata._current_metadata.update_is_empty()

        # Move structural type from rows to top-level.
        outputs_metadata = outputs_metadata.update((), {
            'structural_type': container.ndarray,
        })
        outputs_metadata = outputs_metadata.update((metadata_base.ALL_ELEMENTS,), {
            'structural_type': metadata_base.NO_VALUE,
        })

        return outputs_metadata

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        if issubclass(column_metadata['structural_type'], numpy.ndarray):
            return True

        return False

    def _get_column(self, inputs_metadata: metadata_base.DataMetadata) -> int:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        if self.hyperparams['use_column'] is None:
            use_columns: typing.List[int] = []
        else:
            use_columns = [self.hyperparams['use_column']]

        columns_to_use, _ = base_utils.get_columns_to_use(inputs_metadata, use_columns, self.hyperparams['exclude_columns'], can_use_column)

        if not columns_to_use:
            if use_columns:
                raise ValueError("Specified column cannot be operated on.")
            else:
                raise ValueError("No column found to operate on.")

        assert len(columns_to_use) == 1

        return columns_to_use[0]
