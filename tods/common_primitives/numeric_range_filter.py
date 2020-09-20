import collections
import os
import typing

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives
from common_primitives import dataframe_utils

import pandas as pd  # type: ignore

__all__ = ('NumericRangeFilterPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    column = hyperparams.Hyperparameter[int](
        default=-1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column filter applies to.'
    )
    inclusive = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True when values outside the range are removed, False when values within the range are removed.'
    )
    min = hyperparams.Union[typing.Union[float, None]](
        configuration=collections.OrderedDict(
            float=hyperparams.Hyperparameter[float](0),
            negative_infinity=hyperparams.Constant(None),
        ),
        default='negative_infinity',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Minimum value for filter.'
    )
    max = hyperparams.Union[typing.Union[float, None]](
        configuration=collections.OrderedDict(
            float=hyperparams.Hyperparameter[float](0),
            positive_infinity=hyperparams.Constant(None),
        ),
        default='positive_infinity',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Maximum value for filter.'
    )
    strict = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True when the filter bounds are strict (ie. less than), false then are not (ie. less than equal to).'
    )


class NumericRangeFilterPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which filters rows from a DataFrame based on a numeric range applied to a given column.
    Columns are identified by their index, and the filter itself can be inclusive (values within range are retained)
    or exclusive (values within range are removed).  Boundaries values can be included in the filter (ie. <=) or excluded
    (ie. <).
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8c246c78-3082-4ec9-844e-5c98fcc76f9d',
            'version': '0.1.0',
            'name': "Numeric range filter",
            'python_path': 'd3m.primitives.data_preprocessing.numeric_range_filter.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/numeric_range_filter.py',
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

        if self.hyperparams['min'] is None:
            min = float('inf')
        else:
            min = self.hyperparams['min']

        if self.hyperparams['max'] is None:
            max = float('inf')
        else:
            max = self.hyperparams['max']

        # apply the filter using native dataframe methods
        col_idx = self.hyperparams['column']
        try:
            to_keep: pd.Series
            if self.hyperparams['inclusive']:
                if self.hyperparams['strict']:
                    to_keep = (resource.iloc[:, col_idx].astype(float) > min) & \
                        (resource.iloc[:, col_idx].astype(float) < max)

                else:
                    to_keep = (resource.iloc[:, col_idx].astype(float) >= min) & \
                        (resource.iloc[:, col_idx].astype(float) <= max)
            else:
                if self.hyperparams['strict']:
                    to_keep = (resource.iloc[:, col_idx].astype(float) < min) | \
                        (resource.iloc[:, col_idx].astype(float) > max)
                else:
                    to_keep = (resource.iloc[:, col_idx].astype(float) <= min) | \
                        (resource.iloc[:, col_idx].astype(float) >= max)

            to_keep_indices = resource.loc[to_keep].index

        except ValueError as error:
            raise exceptions.InvalidArgumentValueError(
                "Failure to apply numerical range filter to column {col_idx} of type {type}.".format(
                    col_idx=col_idx,
                    type=resource.iloc[:, col_idx].dtype,
                ),
            ) from error

        # remove dataframe and metadata rows by index
        outputs = dataframe_utils.select_rows(inputs, to_keep_indices)

        return base.CallResult(outputs)
