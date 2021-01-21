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
from pyod.models.ocsvm import OCSVM
from typing import Union
import uuid

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Hyperparamters #######

    kernel = hyperparams.Enumeration[str](
        values=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        default='rbf',
        description='Specifies the kernel type to be used in the algorithm.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    nu = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.5,
        description='An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    degree = hyperparams.Hyperparameter[int](
        default=3,
        description='Degree of the polynomial kernel function (poly). Ignored by all other kernels.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    gamma = hyperparams.Union[Union[float, str]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[float](
                default=0.,
            ),
            ninit=hyperparams.Hyperparameter[str](
                default='auto',
            ),
        ),
        default='ninit',
        description='Kernel coefficient for rbf, poly and sigmoid. If gamma is auto then 1/n_features will be used instead.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    coef0 = hyperparams.Hyperparameter[float](
        default=0.,
        description='Independent term in kernel function. It is only significant in poly and sigmoid.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    tol = hyperparams.Hyperparameter[float](
        default=0.001,
        description='Tolerance for stopping criterion.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    shrinking = hyperparams.UniformBool(
        default=True,
        description='Whether to use the shrinking heuristic.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    cache_size = hyperparams.Hyperparameter[int](
        default=200,
        description='Specify the size of the kernel cache (in MB).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    verbose = hyperparams.UniformBool(
        default=False,
        description='Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    max_iter = hyperparams.Hyperparameter[int](
        default=-1,
        description='Hard limit on iterations within solver, or -1 for no limit.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    pass


class OCSVMPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Wrapper of scikit-learn one-class SVM Class with more functionalities.
    Unsupervised Outlier Detection.
    Estimate the support of a high-dimensional distribution.
    The implementation is based on libsvm.
    See http://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection
    and :cite:`scholkopf2001estimating`.

    Parameters
    ----------
    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    nu : float, optional
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, optional
        Tolerance for stopping criterion.

    shrinking : bool, optional
        Whether to use the shrinking heuristic.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB).

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

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
        "name": "TODS.anomaly_detection_primitives.OCSVMPrimitive",
        "python_path": "d3m.primitives.tods.detection_algorithm.pyod_ocsvm",
        "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods.git']},
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.MARGIN_CLASSIFIER, ],
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "version": "0.0.1",
        "hyperparams_to_tune": ['contamination', 'kernel', 'nu', 'gamma', 'degree'],
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'OCSVMPrimitive'))
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = OCSVM(contamination=hyperparams['contamination'],
                            kernel=hyperparams['kernel'],
                            nu=hyperparams['nu'],
                            degree=hyperparams['degree'],
                            gamma=hyperparams['gamma'],
                            coef0=hyperparams['coef0'],
                            tol=hyperparams['tol'],
                            shrinking=hyperparams['shrinking'],
                            cache_size=hyperparams['cache_size'],
                            verbose=hyperparams['verbose'],
                            max_iter=hyperparams['max_iter'],
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


