from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
# import numpy
import typing

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
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
from .core.KDiscord import KDiscord
import uuid

from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted

from combo.models.score_comb import average, maximization, median, aom, moa
from combo.utils.utility import standardizer

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Hyperparamters #######

    n_neighbors = hyperparams.Hyperparameter[int](
        default=5,
        description='Number of neighbors to use by default for k neighbors queries.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    method = hyperparams.Enumeration[str](
        values=['largest', 'mean', 'median'],
        default='largest',
        description='Combine the distance to k neighbors as the outlier score.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    radius = hyperparams.Hyperparameter[float](
        default=1.0,
        description='Range of parameter space to use by default for `radius_neighbors` queries.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    algorithm = hyperparams.Enumeration[str](
        values=['auto', 'ball_tree', 'kd_tree', 'brute'],
        default='auto',
        description='Algorithm used to compute the nearest neighbors.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    leaf_size = hyperparams.Hyperparameter[int](
        default=30,
        description='Leaf size passed to `BallTree` or `KDTree`. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    metric = hyperparams.Enumeration[str](
        values=['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                'manhattan', 'braycurtis', 'canberra', 'chebyshev',
                'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
                'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
                'sqeuclidean', 'yule'],
        default='minkowski',
        description='metric used for the distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    p = hyperparams.Hyperparameter[int](
        default=2,
        description='Parameter for the Minkowski metric from.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    metric_params = hyperparams.Union[Union[Dict, None]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[Dict](
                default={},
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='Additional keyword arguments for the metric function.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    pass


class KDiscordODetector(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    KDiscord first split multivariate time series into
    subsequences (matrices), and it use kNN outlier detection based on PyOD.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.

    See :cite:`aggarwal2015outlier,zhao2020using` for details.

    Parameters
    ----------
    window_size : int
        The moving window size.

    step_size : int, optional (default=1)
        The displacement for moving window.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.

    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}

        - 'largest': use the distance to the kth neighbor as the outlier score
        - 'mean': use the average of all k neighbors as the outlier score
        - 'median': use the median of the distance to k neighbors as the
          outlier score

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for `radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

        .. deprecated:: 0.74
           ``algorithm`` is deprecated in PyOD 0.7.4 and will not be
           possible in 0.7.6. It has to use BallTree for consistency.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.


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
        "name": "KDiscordODetector",
        "python_path": "d3m.primitives.tods.detection_algorithm.KDiscordODetector",
        "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods.git']},
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LOCAL_OUTLIER_FACTOR, ], #
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "version": "0.0.1",
        "hyperparams_to_tune": ['n_neighbors', 'algorithm', 'leaf_size', 'p', 'contamination',
                                'window_size', 'step_size', 'method', 'radius'],
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'KDiscordODetector')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = KDiscord(window_size=hyperparams['window_size'],
                            contamination=hyperparams['contamination'],
                            step_size=hyperparams['step_size'],
                            n_neighbors=hyperparams['n_neighbors'],
                            method=hyperparams['method'],
                            radius=hyperparams['radius'],
                            algorithm=hyperparams['algorithm'],
                            leaf_size=hyperparams['leaf_size'],
                            metric=hyperparams['metric'],
                            metric_params=hyperparams['metric_params'],
                            p=hyperparams['p'],
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

    def produce_score(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            Outlier score of input DataFrame.
        """
        return super().produce_score(inputs=inputs, timeout=timeout, iterations=iterations)

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

