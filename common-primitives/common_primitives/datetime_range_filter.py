import collections
import os
import typing

from datetime import datetime, timezone
from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives
from common_primitives import dataframe_utils, utils

import pandas as pd  # type: ignore

__all__ = ('DatetimeRangeFilterPrimitive',)

MIN_DATETIME = datetime.min.replace(tzinfo=timezone.utc)
MAX_DATETIME = datetime.min.replace(tzinfo=timezone.utc)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    column = hyperparams.Hyperparameter[int](
        default=-1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column filter applies to.'
    )
    inclusive = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True when values outside the range are removed, False when values within the range are removed.'
    )
    min = hyperparams.Union[typing.Union[datetime, None]](
        configuration=collections.OrderedDict(
            datetime=hyperparams.Hyperparameter[datetime](utils.DEFAULT_DATETIME),
            negative_infinity=hyperparams.Constant(None),
        ),
        default='negative_infinity',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Minimum value for filter. If it is not timezone-aware, it is assumed that it is in UTC timezone.'
    )
    max = hyperparams.Union[typing.Union[datetime, None]](
        configuration=collections.OrderedDict(
            datetime=hyperparams.Hyperparameter[datetime](utils.DEFAULT_DATETIME),
            positive_infinity=hyperparams.Constant(None),
        ),
        default='positive_infinity',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Maximum value for filter. If it is not timezone-aware, it is assumed that it is in UTC timezone.'
    )
    strict = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True when the filter bounds are strict (ie. less than), false then are not (ie. less than equal to).'
    )
    raise_error = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Raise error if the column contains a value which cannot be parsed into a datetime.'
    )


class DatetimeRangeFilterPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which filters rows from a DataFrame based on a datetime range applied to a given column.
    Columns are identified by their index, and the filter itself can be inclusive (values within range are retained)
    or exclusive (values within range are removed).  Boundaries values can be included in the filter (ie. <=) or excluded
    (ie. <).
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '487e5a58-19e9-432c-ac61-fe05c024e42c',
            'version': '0.2.0',
            'name': "Datetime range filter",
            'python_path': 'd3m.primitives.data_preprocessing.datetime_range_filter.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/datetime_range_filter.py',
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

    @classmethod
    def _make_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None:
            return value

        return value.replace(tzinfo=timezone.utc)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # to make sure index matches row indices
        resource = inputs.reset_index(drop=True)

        if self.hyperparams['min'] is None:
            min = datetime.min
        else:
            min = self.hyperparams['min']

        if self.hyperparams['max'] is None:
            max = datetime.max
        else:
            max = self.hyperparams['max']

        min = self._make_aware(min)
        max = self._make_aware(max)

        # apply the filter using native dataframe methods
        col_idx = self.hyperparams['column']
        try:
            parsed_column = resource.iloc[:, col_idx].apply(lambda x: utils.parse_datetime(x))
            if self.hyperparams['raise_error'] and parsed_column.isna().any():
                raise exceptions.InvalidArgumentValueError(
                    "Failure to apply datetime range filter to column {col_idx} of type {type}.".format(
                        col_idx=col_idx,
                        type=resource.iloc[:, col_idx].dtype,
                    ),
                )

            to_keep: pd.Series
            if self.hyperparams['inclusive']:
                if self.hyperparams['strict']:
                    to_keep = (parsed_column > min) & (parsed_column < max)
                else:
                    to_keep = (parsed_column >= min) & (parsed_column <= max)
            else:
                if self.hyperparams['strict']:
                    to_keep = (parsed_column < min) | (parsed_column > max)
                else:
                    to_keep = (parsed_column <= min) | (parsed_column >= max)

            to_keep_indices = resource.loc[to_keep].index

        except (ValueError, OverflowError) as error:
            raise exceptions.InvalidArgumentValueError(
                "Failure to apply datetime range filter to column {col_idx} of type {type}.".format(
                    col_idx=col_idx,
                    type=resource.iloc[:, col_idx].dtype,
                ),
            ) from error

        # remove dataframe and metadata rows by index
        outputs = dataframe_utils.select_rows(inputs, to_keep_indices)

        return base.CallResult(outputs)
