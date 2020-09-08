import os
import typing

import numpy  # type: ignore
import pandas  # type: ignore
from sklearn import model_selection  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams

import common_primitives
from common_primitives import base

__all__ = ('KFoldDatasetSplitPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    number_of_folds = hyperparams.Bounded[int](
        lower=2,
        upper=None,
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Number of folds for k-folds cross-validation.",
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


class KFoldDatasetSplitPrimitive(base.TabularSplitPrimitiveBase[Hyperparams]):
    """
    A primitive which splits a tabular Dataset for k-fold cross-validation.
    """

    __author__ = 'Mingjie Sun <sunmj15@gmail.com>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'bfedaf3a-6dd0-4a83-ad83-3a50fe882bf8',
            'version': '0.1.0',
            'name': "K-fold cross-validation tabular dataset splits",
            'python_path': 'd3m.primitives.evaluation.kfold_dataset_split.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:sunmj15@gmail.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/kfold_split.py',
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
                metadata_base.PrimitiveAlgorithmType.K_FOLD,
                metadata_base.PrimitiveAlgorithmType.CROSS_VALIDATION,
                metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        if self.hyperparams['stratified']:
            if not len(targets.columns):
                raise exceptions.InvalidArgumentValueError("Stratified split is requested, but no target columns found.")

            k_fold = model_selection.StratifiedKFold(
                n_splits=self.hyperparams['number_of_folds'],
                shuffle=self.hyperparams['shuffle'],
                random_state=self._random_state,
            )
        else:
            k_fold = model_selection.KFold(
                n_splits=self.hyperparams['number_of_folds'],
                shuffle=self.hyperparams['shuffle'],
                random_state=self._random_state,
            )

        return list(k_fold.split(attributes, targets))
