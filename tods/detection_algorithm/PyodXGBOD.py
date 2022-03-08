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
from sklearn.utils.validation import check_X_y
from xgboost.sklearn import XGBClassifier # conda install -c conda-forge xgboost
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score
# from numba import njit
from pyod.utils.utility import argmaxn
from pyod.models.base import BaseDetector
from pyod.models.base import BaseDetector
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.loda import LODA

from pyod.utils.utility import check_parameter
from pyod.utils.utility import check_detector
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores

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
from .SODBasePrimitive import Params_SODBase, Hyperparams_SODBase, SupervisedOutlierDetectorBase

from ..common.TODSBasePrimitives import TODSSupervisedLearnerPrimitiveBase, TODSUnsupervisedLearnerPrimitiveBase
from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from pyod.models.xgbod import XGBOD 
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_SODBase):
    ### Add more Attributes ###

    pass


class Hyperparams(Hyperparams_SODBase):
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
class XGBODPrimitive(SupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
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
                        

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        super().set_training_data(inputs=inputs, outputs = outputs)

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
class XGBOD(BaseDetector):
    r"""XGBOD class for outlier detection.
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

    def __init__(self, estimator_list=None, standardization_flag_list=None,
                 max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0,
                 # missing=None,
                 **kwargs):
        super(XGBOD, self).__init__()
        self.estimator_list = estimator_list
        self.standardization_flag_list = standardization_flag_list
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.booster = booster
        self.n_jobs = n_jobs
        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        # self.missing = missing
        self.kwargs = kwargs

    def _init_detectors(self, X):
        """initialize unsupervised detectors if no predefined detectors is
        provided.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The train data
        Returns
        -------
        estimator_list : list of object
            The initialized list of detectors
        standardization_flag_list : list of boolean
            The list of bool flag to indicate whether standardization is needed
        """
        estimator_list = []
        standardization_flag_list = []

        # predefined range of n_neighbors for KNN, AvgKNN, and LOF
        k_range = [1, 3, 5, 10, 20, 30, 40, 50]

        # validate the value of k
        k_range = [k for k in k_range if k < X.shape[0]]

        for k in k_range:
            estimator_list.append(KNN(n_neighbors=k, method='largest'))
            # estimator_list.append(KNN(n_neighbors=k, method='mean'))
            estimator_list.append(LOF(n_neighbors=k))
            # standardization_flag_list.append(True)
            standardization_flag_list.append(True)
            standardization_flag_list.append(True)

        n_bins_range = [5, 10, 15, 20, 25, 30, 50]
        for n_bins in n_bins_range:
            estimator_list.append(HBOS(n_bins=n_bins))
            standardization_flag_list.append(False)

        # predefined range of nu for one-class svm
        nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        for nu in nu_range:
            estimator_list.append(OCSVM(nu=nu))
            standardization_flag_list.append(True)

        # predefined range for number of estimators in isolation forests
        n_range = [10, 20, 50, 70, 100, 150, 200]
        for n in n_range:
            estimator_list.append(
                IForest(n_estimators=n, random_state=self.random_state))
            standardization_flag_list.append(False)

        # # predefined range for number of estimators in LODA
        # n_bins_range = [3, 5, 10, 15, 20, 25, 30, 50]
        # for n_bins in n_bins_range:
        #     estimator_list.append(LODA(n_bins=n_bins))
        #     standardization_flag_list.append(False)

        return estimator_list, standardization_flag_list

    def _validate_estimator(self, X):
        if self.estimator_list is None:
            self.estimator_list, \
            self.standardization_flag_list = self._init_detectors(X)

        # perform standardization for all detectors by default
        if self.standardization_flag_list is None:
            self.standardization_flag_list = [True] * len(self.estimator_list)

        # validate two lists length
        if len(self.estimator_list) != len(self.standardization_flag_list):
            raise ValueError(
                "estimator_list length ({0}) is not equal "
                "to standardization_flag_list length ({1})".format(
                    len(self.estimator_list),
                    len(self.standardization_flag_list)))

        # validate the estimator list is not empty
        check_parameter(len(self.estimator_list), low=1,
                        param_name='number of estimators',
                        include_left=True, include_right=True)

        for estimator in self.estimator_list:
            check_detector(estimator)

        return len(self.estimator_list)

    def _generate_new_features(self, X):
        X_add = np.zeros([X.shape[0], self.n_detector_])

        # keep the standardization scalar for test conversion
        X_norm = self._scalar.transform(X)

        for ind, estimator in enumerate(self.estimator_list):
            if self.standardization_flag_list[ind]:
                X_add[:, ind] = estimator.decision_function(X_norm)

            else:
                X_add[:, ind] = estimator.decision_function(X)
        return X_add

    def fit(self, X, y):
        """Fit the model using X and y as training data.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Training data.
        y : numpy array of shape (n_samples,)
            The ground truth (binary label)
            - 0 : inliers
            - 1 : outliers
        Returns
        -------
        self : object
        """

        # Validate inputs X and y
        X, y = check_X_y(X, y)
        X = check_array(X)
        self._set_n_classes(y)
        self.n_detector_ = self._validate_estimator(X)
        self.X_train_add_ = np.zeros([X.shape[0], self.n_detector_])

        # keep the standardization scalar for test conversion
        X_norm, self._scalar = standardizer(X, keep_scalar=True)

        for ind, estimator in enumerate(self.estimator_list):
            if self.standardization_flag_list[ind]:
                estimator.fit(X_norm)
                self.X_train_add_[:, ind] = estimator.decision_scores_

            else:
                estimator.fit(X)
                self.X_train_add_[:, ind] = estimator.decision_scores_

        # construct the new feature space
        self.X_train_new_ = np.concatenate((X, self.X_train_add_), axis=1)

        # initialize, train, and predict on XGBoost
        self.clf_ = clf = XGBClassifier(max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        silent=self.silent,
                                        objective=self.objective,
                                        booster=self.booster,
                                        n_jobs=self.n_jobs,
                                        nthread=self.nthread,
                                        gamma=self.gamma,
                                        min_child_weight=self.min_child_weight,
                                        max_delta_step=self.max_delta_step,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        colsample_bylevel=self.colsample_bylevel,
                                        reg_alpha=self.reg_alpha,
                                        reg_lambda=self.reg_lambda,
                                        scale_pos_weight=self.scale_pos_weight,
                                        base_score=self.base_score,
                                        random_state=self.random_state,
                                        # missing=self.missing,
                                        **self.kwargs)
        self.clf_.fit(self.X_train_new_, y)
        self.decision_scores_ = self.clf_.predict_proba(
            self.X_train_new_)[:, 1]
        self.labels_ = self.clf_.predict(self.X_train_new_).ravel()

        return self

    def decision_function(self, X):

        check_is_fitted(self, ['clf_', 'decision_scores_',
                               'labels_', '_scalar'])

        X = check_array(X)

        # construct the new feature space
        X_add = self._generate_new_features(X)
        X_new = np.concatenate((X, X_add), axis=1)

        pred_scores = self.clf_.predict_proba(X_new)[:, 1]
        return pred_scores.ravel()

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.
        Calling xgboost `predict` function.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['clf_', 'decision_scores_',
                               'labels_', '_scalar'])

        X = check_array(X)

        # construct the new feature space
        X_add = self._generate_new_features(X)
        X_new = np.concatenate((X, X_add), axis=1)

        pred_scores = self.clf_.predict(X_new)
        return pred_scores.ravel()

    def predict_proba(self, X):
        """Predict the probability of a sample being outlier.
        Calling xgboost `predict_proba` function.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """
        return self.decision_function(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.labels_

    def fit_predict_score(self, X, y, scoring='roc_auc_score'):
        """Fit the detector, predict on samples, and evaluate the model by
        predefined metrics, e.g., ROC.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        scoring : str, optional (default='roc_auc_score')
            Evaluation metric:
            - 'roc_auc_score': ROC score
            - 'prc_n_score': Precision @ rank n score
        Returns
        -------
        score : float
        """

        self.fit(X, y)

        if scoring == 'roc_auc_score':
            score = roc_auc_score(y, self.decision_scores_)
        elif scoring == 'prc_n_score':
            score = precision_n_scores(y, self.decision_scores_)
        else:
            raise NotImplementedError('PyOD built-in scoring only supports '
                                      'ROC and Precision @ rank n')

        print("{metric}: {score}".format(metric=scoring, score=score))

        return score
