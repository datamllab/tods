import os
import uuid
import typing
from collections import OrderedDict

import numpy
import pandas
from sklearn import model_selection

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.base import primitives

from .utils import parse_datetime_to_float

__all__ = ('KFoldTimeSeriesSplitPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    number_of_folds = hyperparams.Bounded[int](
        lower=2,
        upper=None,
        default=5,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'
        ],
        description="Number of folds for k-folds cross-validation.",
    )
    number_of_window_folds = hyperparams.Union[typing.Union[int, None]](
        configuration=OrderedDict(
            fixed=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=1,
                description="Number of folds in train set (window). These folds come directly "
                "before test set (streaming window).",
            ),
            all_records=hyperparams.Constant(
                default=None,
                description="Number of folds in train set (window) = maximum number possible.",
            ),
        ),
        default='all_records',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'
        ],
        description="Maximum size for a single training set.",
    )
    time_column_index = hyperparams.Union[typing.Union[int, None]](
        configuration=OrderedDict(
            fixed=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=1,
                description="Specific column that contains the time index",
            ),
            one_column=hyperparams.Constant(
                default=None,
                description="Only one column contains a time index. "
                "It is detected automatically using semantic types.",
            ),
        ),
        default='one_column',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'
        ],
        description="Column index to use as datetime index. "
        "If None, it is required that only one column with time column role semantic type is "
        "present and otherwise an exception is raised. "
        "If column index specified is not a datetime column an exception is"
        "also raised.",
    )
    fuzzy_time_parsing = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter'
        ],
        description="Use fuzzy time parsing.",
    )


class KFoldTimeSeriesSplitPrimitive(primitives.TabularSplitPrimitiveBase[Hyperparams]):
    """
    A primitive which splits a tabular time-series Dataset for k-fold cross-validation.

    Primitive sorts the time column so care should be taken to assure sorting of a
    column is reasonable. E.g., if column is not numeric but of string structural type,
    strings should be formatted so that sorting by them also sorts by time.
    """

    __author__ = 'Distil'
    __version__ = '0.3.0'
    __contact__ = 'mailto:jeffrey.gleason@yonder.co'

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '002f9ad1-46e3-40f4-89ed-eeffbb3a102b',
            'version': __version__,
            'name': "K-fold cross-validation timeseries dataset splits",
            'python_path': 'd3m.primitives.tods.evaluation.kfold_time_series_split',
            'source': {
                'name': 'DATALab@Texas A&M University',
                'contact': __contact__,
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/kfold_split_timeseries.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.K_FOLD,
                metadata_base.PrimitiveAlgorithmType.CROSS_VALIDATION,
                metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        time_column_indices = dataset.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/Time'], at=(main_resource_id,))
        attribute_column_indices = dataset.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/Attribute'], at=(main_resource_id,))

        # We want only time columns which are also attributes.
        time_column_indices = [time_column_index for time_column_index in time_column_indices if time_column_index in attribute_column_indices]

        if self.hyperparams['time_column_index'] is None:
            if len(time_column_indices) != 1:
                raise exceptions.InvalidArgumentValueError(
                    "If \"time_column_index\" hyper-parameter is \"None\", it is required that exactly one column with time column role semantic type is present.",
                )
            else:
                # We know it exists because "time_column_indices" is a subset of "attribute_column_indices".
                time_column_index = attribute_column_indices.index(
                    time_column_indices[0],
                )
        else:
            if self.hyperparams['time_column_index'] not in time_column_indices:
                raise exceptions.InvalidArgumentValueError(
                    "Time column index specified does not have a time column role semantic type.",
                )
            else:
                time_column_index = attribute_column_indices.index(
                    self.hyperparams['time_column_index'],
                )

        # We first reset index.
        attributes = attributes.reset_index(drop=True)

        # Then convert datetime column to consistent datetime representation
        attributes.insert(
            loc=0,
            column=uuid.uuid4(),  # use uuid to ensure we are inserting a new column name
            value=self._parse_time_data(
                attributes, time_column_index, self.hyperparams['fuzzy_time_parsing'],
            ),
        )

        # Then sort dataframe by new datetime column. Index contains original row order.
        attributes = attributes.sort_values(by=attributes.columns[0])

        # Remove datetime representation used for sorting (primitives might choose to parse this str col differently).
        attributes = attributes.drop(attributes.columns[0], axis=1)

        max_train_size: typing.Optional[int] = None
        if self.hyperparams['number_of_window_folds'] is not None:
            max_train_size = int(attributes.shape[0] * self.hyperparams['number_of_window_folds'] / self.hyperparams['number_of_folds'])

        k_fold = model_selection.TimeSeriesSplit(
            n_splits=self.hyperparams['number_of_folds'],
            max_train_size=max_train_size
        )

        # We sorted "attributes" so we have to map indices on sorted "attributes" back to original
        # indices. We do that by using DataFrame's index which contains original row order.
        return [
            (
                numpy.array([attributes.index[val] for val in train]),
                numpy.array([attributes.index[val] for val in test]),
            )
            for train, test in k_fold.split(attributes)
        ]

    @classmethod
    def _parse_time_data(cls, inputs: container.DataFrame, column_index: metadata_base.SimpleSelectorSegment, fuzzy: bool) -> typing.List[float]:
        return [
            parse_datetime_to_float(value, fuzzy=fuzzy)
            for value in inputs.iloc[:, column_index]
        ]
