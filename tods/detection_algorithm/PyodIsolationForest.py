from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
# from numba import njit
from pyod.utils.utility import argmaxn

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from pyod.models.iforest import IForest
from typing import Union
import uuid

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Hyperparamters #######

    n_estimators = hyperparams.Hyperparameter[int](
        default=100,
        description='The number of base estimators in the ensemble.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    max_samples = hyperparams.Enumeration[str](
        values=['auto', 'int', 'float'],
        default='auto',  # 'box-cox', #
        description='The number of samples to draw from X to train each base estimator.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    max_features = hyperparams.Hyperparameter[float](
        default=1.,
        description='The number of features to draw from X to train each base estimator.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    bootstrap = hyperparams.UniformBool(
        default=False,
        description='If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement is performed.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    behaviour = hyperparams.Enumeration[str](
        values=['old', 'new'],
        default='new',
        description='Refer to https://github.com/yzhao062/pyod/blob/master/pyod/models/iforest.py.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    random_state = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=0,
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='the seed used by the random number generator.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    verbose = hyperparams.Hyperparameter[int](
        default=0,
        description='Controls the verbosity of the tree building process.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    pass


class IsolationForest(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):

    """
    Wrapper of Pyod Isolation Forest with more functionalities.
    The IsolationForest 'isolates' observations by randomly selecting a
    feature and then randomly selecting a split value between the maximum and
    minimum values of the selected feature.
    See :cite:`liu2008isolation,liu2012isolation` for details.
    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.
    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    behaviour : str, default='old'
        Behaviour of the ``decision_function`` which can be either 'old' or
        'new'. Passing ``behaviour='new'`` makes the ``decision_function``
        change to match other anomaly detection algorithm API which will be
        the default behaviour in the future. As explained in details in the
        ``offset_`` attribute documentation, the ``decision_function`` becomes
        dependent on the contamination parameter, in such a way that 0 becomes
        its natural threshold to detect outliers.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "name": "TODS.anomaly_detection_primitives.IsolationForest",
        "python_path": "d3m.primitives.tods.detection_algorithm.pyod_iforest",
        "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods.git']},
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.ISOLATION_FOREST, ],
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "version": "0.0.1",
        "hyperparams_to_tune": ['n_estimators', 'contamination'],
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'IsolationForest'))
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = IForest(contamination=hyperparams['contamination'],
                            n_estimators=hyperparams['n_estimators'],
                            max_samples=hyperparams['max_samples'],
                            max_features=hyperparams['max_features'],
                            bootstrap=hyperparams['bootstrap'],
                            behaviour=hyperparams['behaviour'],
                            random_state=hyperparams['random_state'],
                            verbose=hyperparams['verbose'],
                            )


        return

    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        super().set_training_data(inputs=inputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        return super().fit()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            1 marks Outliers, 0 marks normal.
        """
        return super().produce(inputs=inputs, timeout=timeout, iterations=iterations)

    def get_params(self) -> Params:
        """
        Return parameters.
        Args:
            None

        Returns:
            class Params
        """
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters for outlier detection.
        Args:
            params: class Params

        Returns:
            None
        """
        super().set_params(params=params)


