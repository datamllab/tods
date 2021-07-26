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
import uuid

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from pyod.models.xgbod import XGBOD 
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ### Add more Attributes ###

    pass


class Hyperparams(Hyperparams_ODBase):
    ### Add more Hyperparamters ###
    estimator_list = hyperparams.Union[Union[int, None]]( 
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=1, # {},
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='The list of pyod detectors passed in for unsupervised learning.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

    standardization_flag_list = hyperparams.Union[Union[int, None]]( 
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=1, # {},
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='The list of boolean flags for indicating whether to perform standardization for each detector.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

    max_depth = hyperparams.Hyperparameter[int]( 
        default=3,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Maximum tree depth for base learners.',
    )

    # learning_rate = hyperparams.Hyperparameter[float]( 
    #     default=0.1,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    #     description='Boosting learning rate (xgb "eta").',
    # )

    learning_rate = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.1,
        description='Boosting learning rate (xgb "eta").',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    n_estimators = hyperparams.Hyperparameter[int]( 
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Number of boosted trees to fit.',
    )
    
    silent = hyperparams.UniformBool(
        default=True,
        description='Whether to print messages while running boosting.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    booster = hyperparams.Enumeration[str]( 
        values=['gbtree', 'gblinear', 'dart'],
        default='gbtree',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Specify which booster to use: gbtree, gblinear or dart.',
    )

    n_jobs = hyperparams.Hyperparameter[int]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Number of parallel threads used to run xgboost.  (replaces ``nthread``).',
    )

    gamma = hyperparams.Hyperparameter[float]( 
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Minimum loss reduction required to make a further partition on a leaf node of the tree.',
    )

    min_child_weight = hyperparams.Hyperparameter[int]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Minimum sum of instance weight(hessian) needed in a child.',
    )

    max_delta_step = hyperparams.Hyperparameter[int]( 
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Maximum delta step we allow each tree weight estimation to be.',
    )

    subsample = hyperparams.Hyperparameter[float]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Subsample ratio of the training instance.',
    )

    colsample_bytree = hyperparams.Hyperparameter[float]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Subsample ratio of columns when constructing each tree.',
    )

    colsample_bylevel = hyperparams.Hyperparameter[float]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Subsample ratio of columns for each split, in each level.',
    )

    reg_alpha = hyperparams.Hyperparameter[float]( 
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='L1 regularization term on weights.',
    )

    reg_lambda = hyperparams.Hyperparameter[float]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='L2 regularization term on weights.',
    )

    scale_pos_weight = hyperparams.Hyperparameter[float]( 
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Balancing of positive and negative weights.',
    )

    base_score = hyperparams.Hyperparameter[float]( 
        default=0.5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='The initial prediction score of all instances, global bias.',
    )

    # random_state = hyperparams.Hyperparameter[int]( #controlled by d3m
    #     default=0,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    #     description='Random number seed.  (replaces seed).',
    # )

### Name of your algorithm ###
class XGBODPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    XGBOD class for outlier detection.
    It first uses the passed in unsupervised outlier detectors to extract
    richer representation of the data and then concatenates the newly
    generated features to the original feature for constructing the augmented
    feature space. An XGBoost classifier is then applied on this augmented
    feature space. Read more in the :cite:`zhao2018xgbod`.
    Parameters
    ----------
    estimator_list : list, optional (default=None)
        The list of pyod detectors passed in for unsupervised learning
    standardization_flag_list : list, optional (default=None)
        The list of boolean flags for indicating whether to perform
        standardization for each detector.
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    n_estimators : int
        Number of boosted trees to fit.
    silent : bool
        Whether to print messages while running boosting.
    objective : string or callable
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    booster : string
        Specify which booster to use: gbtree, gblinear or dart.
    n_jobs : int
        Number of parallel threads used to run xgboost.  (replaces ``nthread``)
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float
        Subsample ratio of columns for each split, in each level.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights.
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights.
    scale_pos_weight : float
        Balancing of positive and negative weights.
    base_score:
        The initial prediction score of all instances, global bias.
    random_state : int
        Random number seed.  (replaces seed)
    # missing : float, optional
    #     Value in the data which needs to be present as a missing value. If
    #     None, defaults to np.nan.
    importance_type: string, default "gain"
        The feature importance type for the ``feature_importances_``
        property: either "gain",
        "weight", "cover", "total_gain" or "total_cover".
    \*\*kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of
        parameters can be found here:
        https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst.
        Attempting to set a parameter via the constructor args and \*\*kwargs
        dict simultaneously will result in a TypeError.
        Note: \*\*kwargs is unsupported by scikit-learn. We do not
        guarantee that parameters passed via this argument will interact
        properly with scikit-learn.
    Attributes
    ----------
    n_detector_ : int
        The number of unsupervised of detectors used.
    clf_ : object
        The XGBoost classifier.
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    ### Modify the metadata ###
    #__author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({
         "__author__": "DATA Lab at Texas A&M University",
         "name": "XGBOD Primitive",
         "python_path": "d3m.primitives.tods.detection_algorithm.pyod_xgbod",
         "source": {
             'name': 'DATA Lab at Texas A&M University', 
             'contact': 'mailto:khlai037@tamu.edu', 
         },
         "hyperparams_to_tune": ['estimator_list', 'standardization_flag_list', 'max_depth', 'learning_rate','n_estimators','silent','booster','n_jobs','gamma','min_child_weight','max_delta_step','subsample','colsample_bytree','colsample_bylevel','reg_alpha','reg_lambda','scale_pos_weight','base_score'],
         "version": "0.0.1",
         "algorithm_types": [
             metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE
         ],
         "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
	 'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'XGBODPrimitive')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        ### Initialize your algorithm ###
        self._clf = XGBOD(estimator_list=hyperparams['estimator_list'],
                        standardization_flag_list=hyperparams['standardization_flag_list'],
                        max_depth=hyperparams['max_depth'],
                        learning_rate=hyperparams['learning_rate'],
                        n_estimators=hyperparams['n_estimators'],
                        silent=hyperparams['silent'],
                        objective="binary:logistic",
                        booster=hyperparams['booster'],
                        n_jobs=hyperparams['n_jobs'],
                        nthread=None,
                        gamma=hyperparams['gamma'],
                        min_child_weight=hyperparams['min_child_weight'],
                        max_delta_step=hyperparams['max_delta_step'],
                        subsample=hyperparams['subsample'],
                        colsample_bytree=hyperparams['colsample_bytree'],
                        colsample_bylevel=hyperparams['colsample_bylevel'],
                        reg_alpha=hyperparams['reg_alpha'],
                        reg_lambda=hyperparams['reg_lambda'],
                        scale_pos_weight=hyperparams['scale_pos_weight'],
                        base_score=hyperparams['base_score'],
                        # random_state=hyperparams['random_state'],
                        )
                        

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



### The Implementation of your algorithm ###
class DetectionAlgorithm(BaseDetector):
    """
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

    def __init__():
        pass       

    def fit():
        pass

    def decision_function(self):
        pass
