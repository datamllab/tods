from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import pandas as pd

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
import uuid

from d3m import container, utils as d3m_utils
from .core.CollectiveBase import CollectiveBaseDetector
from .core.utility import get_sub_matrices

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
import stumpy

from sklearn.preprocessing import MinMaxScaler
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe
from tods.utils import construct_primitive_metadata


class Params(Params_ODBase):
    ######## Add more Attributes #######
    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Attributes #######
    #pass
    window_size = hyperparams.Hyperparameter[int](
        default=3,
        description='The moving window size.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

class MP(CollectiveBaseDetector):
    """
    This is the class for matrix profile function
    """
    def __init__(self, window_size, step_size, contamination):
        self._window_size = window_size
        self._step_size = step_size
        self.contamination = contamination
        return

    def _get_right_inds(self, data):
        right_inds = []
        for row in data[1]:
            right_inds.append(row+self._window_size-1)
        right_inds = pd.DataFrame(right_inds)
        data = pd.concat([data,right_inds], axis=1)
        data.columns = range(0,len(data.columns))
        return data

    def fit(self, X):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        sub_matrices, self.left_inds_, self.right_inds_ = get_sub_matrices(
            X,
            window_size=self._window_size,
            step=self._step_size,
            return_numpy=True,
            flatten=True)
        sub_matrices = sub_matrices[:-1, :]
        #self.left_inds_ = self.left_inds_[:-1]
        #self.right_inds_ = self.right_inds_[:-1]
        matrix_profile, matrix_profile_indices = stumpy.mstump(X.transpose(), m = self._window_size)
        blank = X.shape[0] - matrix_profile.shape[0]
        for i in range(blank):
            matrix_profile = np.append(matrix_profile, [matrix_profile[-1]], axis=0)
        #matrix_profile, matrix_profile_indices = stumpy.mstump(data, m = self._window_size)

        #left_inds_ = numpy.arange(0, len(matrix_profile), self._step_size)
        #right_inds_ = left_inds_ + self._window_size
        #right_inds_[right_inds_ > len(matrix_profile)] = len(matrix_profile)
        #left_inds_ = np.array([left_inds_]).transpose()
        #right_inds_ = np.array([right_inds_]).transpose()
        
        # apply min-max scaling
        scaler = MinMaxScaler()
        scaler = scaler.fit(matrix_profile)
        matrix_profile = scaler.transform(matrix_profile)

        # sum over the dimension with normalized MP value
        if len(matrix_profile.shape) > 1 or matrix_profile.shape[1] > 1:
            matrix_profile = np.sum(matrix_profile, axis=1)
        self.decision_scores_ = matrix_profile
        self._process_decision_scores()
        return self

    def decision_function(self, X):

        """
        Args:
            data: dataframe column
        Returns:
            nparray
        """
        """
        #only keep first two columns of MP results, the second column is left index, use windowsize to get right index
        transformed_columns=utils.pandas.DataFrame()
        for col in data.transpose(): #data.reshape(1,len(data)):
            output = stumpy.stump(col, m = self._window_size)
            output = pd.DataFrame(output)
            output=output.drop(columns=[2,3])
            output = self._get_right_inds(output)
            transformed_columns=pd.concat([transformed_columns,output], axis=1)
        return transformed_columns
        """
        sub_matrices, left_inds_, right_inds_ = get_sub_matrices(
            X,
            window_size=self._window_size,
            step=self._step_size,
            return_numpy=True,
            flatten=True)
        sub_matrices = sub_matrices[:-1, :]
        matrix_profile, matrix_profile_indices = stumpy.mstump(X.transpose(), m = self._window_size)
        blank = X.shape[0] - matrix_profile.shape[0]
        for i in range(blank):
            matrix_profile = np.append(matrix_profile, [matrix_profile[-1]], axis=0)
        #matrix_profile, matrix_profile_indices = stumpy.mstump(data, m = self._window_size)

        #left_inds_ = numpy.arange(0, len(matrix_profile), self._step_size)
        #right_inds_ = left_inds_ + self._window_size
        #right_inds_[right_inds_ > len(matrix_profile)] = len(matrix_profile)
        #left_inds_ = np.array([left_inds_]).transpose()
        #right_inds_ = np.array([right_inds_]).transpose()
        
        # apply min-max scaling
        scaler = MinMaxScaler()
        scaler = scaler.fit(matrix_profile)
        matrix_profile = scaler.transform(matrix_profile)

        # sum over the dimension with normalized MP value
        if len(matrix_profile.shape) > 1 or matrix_profile.shape[1] > 1:
            matrix_profile = np.sum(matrix_profile, axis=1)

        return matrix_profile, left_inds_, right_inds_
        
class MatrixProfilePrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that performs matrix profile on a DataFrame using Stumpy package
    Stumpy documentation: https://stumpy.readthedocs.io/en/latest/index.html

Parameters
----------
    T_A : ndarray
        The time series or sequence for which to compute the matrix profile
    m : int
        Window size
    T_B : ndarray
        The time series or sequence that contain your query subsequences
        of interest. Default is `None` which corresponds to a self-join.
    ignore_trivial : bool
        Set to `True` if this is a self-join. Otherwise, for AB-join, set this
        to `False`. Default is `True`.
      
.. dropdown:: Returns  
    
    out : ndarray
        The first column consists of the matrix profile, the second column
        consists of the matrix profile indices, the third column consists of
        the left matrix profile indices, and the fourth column consists of
        the right matrix profile indices.
    
    """

    metadata = construct_primitive_metadata(module='detection_algorithm', name='matrix_profile', id='MatrixProfilePrimitive', primitive_family='anomaly_detect', hyperparams=['window_size'], description='Matrix Profile')


    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = MP(window_size=hyperparams['window_size'], step_size=hyperparams['step_size'], contamination=hyperparams['contamination'])

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

    def get_params(self) -> Params:        # pragma: no cover
        """
        Return parameters.
        Args:
            None
        Returns:
            class Params
        """
        return super().get_params()

    def set_params(self, *, params: Params) -> None:    # pragma: no cover
        """
        Set parameters for outlier detection.
        Args:
            params: class Params
        Returns:
            None
        """
        super().set_params(params=params)




