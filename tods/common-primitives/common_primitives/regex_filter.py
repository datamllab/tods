import os
import re

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives
from common_primitives import dataframe_utils

__all__ = ('RegexFilterPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    column = hyperparams.Hyperparameter[int](
        default=-1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column filter applies to.',
    )
    inclusive = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True when values that match the pattern are removed, False when they are removed.',
    )
    regex = hyperparams.Hyperparameter[str](
        default="",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='A python regular expression string to act as a filter.',
    )


class RegexFilterPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which filters rows from a DataFrame based on a regex applied to a given column.
    Columns are identified by index.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'cf73bb3d-170b-4ba9-9ead-3dd4b4524b61',
            'version': '0.1.0',
            'name': "Regex dataset filter",
            'python_path': 'd3m.primitives.data_preprocessing.regex_filter.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/regex_filter.py',
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
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # to make sure index matches row indices
        resource = inputs.reset_index(drop=True)

        try:
            # apply the filter
            pattern = re.compile(self.hyperparams['regex'])
            matched = resource.iloc[:, self.hyperparams['column']].astype(str).str.contains(pattern)
            to_keep = matched if self.hyperparams['inclusive'] else ~matched

            to_keep_indices = resource.loc[to_keep].index

        except re.error as error:
            raise exceptions.InvalidArgumentValueError("Invalid regex: {regex}".format(regex=self.hyperparams['regex'])) from error

        # remove dataframe and metadata rows by index
        outputs = dataframe_utils.select_rows(inputs, to_keep_indices)

        return base.CallResult(outputs)
