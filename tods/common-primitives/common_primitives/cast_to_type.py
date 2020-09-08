import os
import typing

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('CastToTypePrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    type_to_cast = hyperparams.Enumeration[str](
        values=['str', 'float'],
        default='str',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be cast to the type, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )


class CastToTypePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which casts all columns it can cast (by default, controlled by ``use_columns``,
    ``exclude_columns``) of an input DataFrame to a given structural type (dtype).
    It removes columns which are not cast.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'eb5fe752-f22a-4090-948b-aafcef203bf5',
            'version': '0.2.0',
            'name': "Casts DataFrame",
            'python_path': 'd3m.primitives.data_transformation.cast_to_type.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/cast_to_type.py',
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

    _type_map = {
        'str': str,
        'float': float,
    }

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        type_to_cast = self._type_map[self.hyperparams['type_to_cast']]

        columns_to_use = self._get_columns(inputs.metadata, type_to_cast)

        outputs = inputs.iloc[:, list(columns_to_use)].astype(type_to_cast)
        outputs_metadata = inputs.metadata.select_columns(columns_to_use)

        outputs_metadata = outputs_metadata.update((metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS), {
            'structural_type': type_to_cast,
        })

        outputs.metadata = outputs_metadata

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int, type_to_cast: type) -> bool:
        if type_to_cast == str:
            # TODO: Anything can be converted to string, but is it meaningful (Python string description of object might not be)? Should we limit what can be cast this way?
            return True
        else:
            column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            structural_type = column_metadata.get('structural_type', None)

            if structural_type is None:
                return False

            return d3m_utils.is_numeric(structural_type)

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata, type_to_cast: type) -> typing.Sequence[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index, type_to_cast)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        if not columns_to_use:
            raise ValueError("No columns to be cast to type '{type}'.".format(type=type_to_cast))

        # We prefer if all columns could be cast, not just specified columns,
        # so we warn always when there are columns which cannot be produced.
        elif columns_not_to_use:
            self.logger.warning("Not all columns can be cast to type '%(type)s'. Skipping columns: %(columns)s", {
                'type': type_to_cast,
                'columns': columns_not_to_use,
            })

        return columns_to_use
