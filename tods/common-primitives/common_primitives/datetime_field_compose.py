import os

from dateutil import parser
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('DatetimeFieldComposePrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to use when composing a datetime field.",
    )
    join_char = hyperparams.Hyperparameter[str](
        default="-",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='A string used to join fields prior to parsing a datetime.',
    )
    output_name = hyperparams.Hyperparameter[str](
        default="__date",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The name to use for the new parsed datetime field.',
    )


class DatetimeFieldComposePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which composes fields into a new single datetime field.

    The primitve joins the columns (identified in the columns hyperparam) in order and then parses
    the resulting string as a datetime. The value is stored in a new column.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '73d79f46-1bea-4858-a061-a2d1cfc5f122',
            'version': '0.1.0',
            'name': "Datetime Field Compose",
            'python_path': 'd3m.primitives.data_transformation.datetime_field_compose.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/datetime_field_compose.py',
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
        inputs_clone = inputs.copy()
        columns = self.hyperparams['columns']
        output_name = self.hyperparams['output_name']
        join_char = self.hyperparams['join_char']

        new_col = inputs_clone.iloc[:, list(columns)].apply(lambda x: parser.parse(join_char.join(x)), axis=1)
        new_col_index = len(inputs_clone.columns)
        inputs_clone.insert(new_col_index, output_name, new_col)
        inputs_clone.metadata = inputs_clone.metadata.generate(inputs_clone)
        inputs_clone.metadata = inputs_clone.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, new_col_index), 'http://schema.org/DateTime')
        inputs_clone.metadata = inputs_clone.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, new_col_index), 'https://metadata.datadrivendiscovery.org/types/Time')

        return base.CallResult(inputs_clone)
