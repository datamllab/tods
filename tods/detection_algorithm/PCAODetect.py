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
from .core.PCA import PCA
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

    svd_solver = hyperparams.Enumeration[str](
        values=['auto', 'full', 'arpack', 'randomized'],
        default='auto',
        description='Algorithm of solver.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    n_components = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=1, # {},
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='Number of components to keep. It should be smaller than the window_size.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

    #     hyperparams.Hyperparameter[int](
    #     default=1,
    #     description='Number of components to keep. It should be smaller than the window_size.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )

    n_selected_components = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=1, # {},
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='Number of selected principal components for calculating the outlier scores. It is not necessarily equal to the total number of the principal components. If not set, use all principal components.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

    tol = hyperparams.Hyperparameter[float](
        default=0.,
        description='Tolerance for singular values computed by svd_solver == `arpack`.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    iterated_power = hyperparams.Union[Union[int, str]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=1, # {},
            ),
            ninit=hyperparams.Hyperparameter[str](
                default='auto',
            ),
        ),
        default='ninit',
        description='Number of iterations for the power method computed by svd_solver == `randomized`.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
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
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

    whiten = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="If True, the eigenvalues are used in score computation. The eigenvectors with small eigenvalues comes with more importance in outlier score calculation.",
    )

    standardization = hyperparams.UniformBool(
            default=True,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="If True, perform standardization first to convert data to zero mean and unit variance.",
    )

    pass


class PCAODetector(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    PCA-based outlier detection with both univariate and multivariate
    time series data. TS data will be first transformed to tabular format.
    For univariate data, it will be in shape of [valid_length, window_size].
    for multivariate data with d sequences, it will be in the shape of
    [valid_length, window_size].

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

    n_components : int, float, None or string
        Number of components to keep. It should be smaller than the window_size.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.

    n_selected_components : int, optional (default=None)
        Number of selected principal components
        for calculating the outlier scores. It is not necessarily equal to
        the total number of the principal components. If not set, use
        all principal components.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    weighted : bool, optional (default=True)
        If True, the eigenvalues are used in score computation.
        The eigenvectors with small eigenvalues comes with more importance
        in outlier score calculation.

    standardization : bool, optional (default=True)
        If True, perform standardization first to convert
        data to zero mean and unit variance.
        See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

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
        "name": "PCAODetector",
        "python_path": "d3m.primitives.tods.detection_algorithm.PCAODetector",
        "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods.git']},
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LOCAL_OUTLIER_FACTOR, ], #
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "version": "0.0.1",
        "hyperparams_to_tune": ['n_components', 'n_selected_components', 'contamination',
                                'whiten', 'svd_solver', 'tol', 'iterated_power', 'random_state',
                                'standardization'],
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'PCAODetector')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = PCA(window_size=hyperparams['window_size'],
                        contamination=hyperparams['contamination'],
                        n_components=hyperparams['n_components'],
                        n_selected_components=hyperparams['n_selected_components'],
                        whiten=hyperparams['whiten'],
                        svd_solver=hyperparams['svd_solver'],
                        tol=hyperparams['tol'],
                        iterated_power=hyperparams['iterated_power'],
                        random_state=hyperparams['random_state'],
                        standardization=hyperparams['standardization'],
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


