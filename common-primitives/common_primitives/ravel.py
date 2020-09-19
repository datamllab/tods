import os
import typing

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('RavelAsRowPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    order = hyperparams.Enumeration(
        values=['row-major', 'column-major'],
        default='row-major',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="\"row-major\" means to index the elements in row-major, with the last axis index changing fastest, back to the first axis index changing slowest. "
                    "\"column-major\" means to index the elements in column-major, with the first index changing fastest, and the last index changing slowest.",
    )


class RavelAsRowPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which ravels all values in the DataFrame into one row.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'fe87544d-ef93-48d0-b420-76768d351f39',
            'version': '0.1.0',
            'name': "Ravel a DataFrame into a row",
            'python_path': 'd3m.primitives.data_transformation.ravel.DataFrameRowCommon',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/ravel.py',
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

    @base.singleton
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        output_values: typing.List[typing.List] = [[]]
        output_columns = []

        rows_length, columns_length = inputs.shape

        if self.hyperparams['order'] == 'row-major':
            for row in inputs.itertuples(index=False, name=None):
                for column_index, value in enumerate(row):
                    output_values[0].append(value)
                    output_columns.append(inputs.columns[column_index])

        elif self.hyperparams['order'] == 'column-major':
            for column_index in range(columns_length):
                for value in inputs.iloc[:, column_index]:
                    output_values[0].append(value)
                    output_columns.append(inputs.columns[column_index])

        else:
            raise exceptions.InvalidArgumentValueError(f"Invalid \"order\" hyper-parameter value \"{self.hyperparams['order']}\".")

        assert len(output_values[0]) == len(output_columns)
        assert len(output_values[0]) == rows_length * columns_length

        outputs = container.DataFrame(output_values, columns=output_columns, metadata=inputs.metadata.query(()), generate_metadata=False)
        outputs.metadata = outputs.metadata.update((), {
            'dimension': {
                'length': 1,
            },
        })
        outputs.metadata = outputs.metadata.update((metadata_base.ALL_ELEMENTS,), {
            'dimension': {
                'length': len(output_columns),
            },
        })
        outputs.metadata = outputs.metadata.set_table_metadata()

        if self.hyperparams['order'] == 'row-major':
            for index in range(len(output_columns)):
                row_index = int(index / columns_length)
                column_index = int(index % columns_length)

                outputs.metadata = outputs.metadata.update(
                    (0, index),
                    inputs.metadata.query((row_index, column_index))
                )

        elif self.hyperparams['order'] == 'column-major':
            for index in range(len(output_columns)):
                row_index = int(index % rows_length)
                column_index = int(index / rows_length)

                outputs.metadata = outputs.metadata.update(
                    (0, index),
                    inputs.metadata.query((row_index, column_index))
                )

        else:
            assert False

        return base.CallResult(outputs)
