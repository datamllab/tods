import os
import typing

import numpy
import pandas
from sklearn import model_selection

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.base import primitives

from tods.utils import construct_primitive_metadata
__all__ = ('TrainScoreDatasetSplitPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    train_score_ratio = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.75,
        upper_inclusive=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The ratio between the train and score data and represents the proportion of the Dataset to include in the train split. The rest is included in the score split.",
    )
    stratified = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Do stratified folds. The folds are made by preserving the percentage of samples for each class.",
    )
    shuffle = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether to shuffle the data before splitting into batches.",
    )
    delete_recursive = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Delete rows in other resources/tables which are not needed for rows left in the dataset entry point resource/table.",
    )


class TrainScoreDatasetSplitPrimitive(primitives.TabularSplitPrimitiveBase[Hyperparams]):
    """
    A primitive which splits a tabular Dataset into random train and score subsets.
    """

    metadata = construct_primitive_metadata(module='detection_algorithm', name='train_score_dataset_split', id='3fcc6dc4-6681-4c86-948e-066d14e7d803', primitive_family='evaluation', algorithm= ['holdout','data_split'], description='Train-score tabular dataset splits')
    

    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        if self.hyperparams['stratified'] and not len(targets.columns):
            raise exceptions.InvalidArgumentValueError("Stratified split is requested, but no target columns found.")

        train_data, score_data = model_selection.train_test_split(
            numpy.arange(len(attributes)),
            test_size=None,
            train_size=self.hyperparams['train_score_ratio'],
            random_state=self._random_state,
            shuffle=self.hyperparams['shuffle'],
            stratify=targets if self.hyperparams['stratified'] else None,
        )

        return [(train_data, score_data)]
