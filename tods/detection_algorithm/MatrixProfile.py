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

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
import stumpy
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe



class Params(Params_ODBase):
	######## Add more Attributes #######
	pass


class Hyperparams(Hyperparams_ODBase):
	######## Add more Attributes #######
	pass

class MP:
	"""
	This is the class for matrix profile function
	"""
	def __init__(self, window_size):
		self._window_size = window_size
		return

	def fit(self, X, y=None):
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

		# validate inputs X and y (optional)
		# X = check_array(X)
		# self._set_n_classes(y)
		# self.decision_scores_ = self.decision_function(X)
		# self._process_decision_scores()
		
		return self

	def _get_right_inds(self, data):
		right_inds = []
		for row in data[1]:
			right_inds.append(row+self._window_size-1)
		right_inds = pd.DataFrame(right_inds)
		data = pd.concat([data,right_inds], axis=1)
		data.columns = range(0,len(data.columns))
		return data

	def produce(self, data):

		"""

		Args:
			data: dataframe column
		Returns:
			nparray

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

	def predict(self, data):
<<<<<<< HEAD
		return self.produce(data)	
	
	def decision_function(self, data):
=======
>>>>>>> 9ba3c17c4fed739483897eadb775a09610badbcb
		return self.produce(data)
		
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
    	Returnsfdsf
    	-------
    	out : ndarray
    	    The first column consists of the matrix profile, the second column
    	    consists of the matrix profile indices, the third column consists of
    	    the left matrix profile indices, and the fourth column consists of
    	    the right matrix profile indices.
	
	"""

	metadata = metadata_base.PrimitiveMetadata({
		'__author__': "DATA Lab @Texas A&M University",
		'name': "Matrix Profile",
		#'python_path': 'd3m.primitives.tods.feature_analysis.matrix_profile',
		'python_path': 'd3m.primitives.tods.detection_algorithm.matrix_profile',
		'source': {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods/-/blob/Yile/anomaly-primitives/anomaly_primitives/MatrixProfile.py']},
		'algorithm_types': [metadata_base.PrimitiveAlgorithmType.MATRIX_PROFILE,], 
		'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
		'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'MatrixProfilePrimitive')),
		'hyperparams_to_tune': ['window_size'],
		'version': '0.0.2',		
		})


	def __init__(self, *,
				 hyperparams: Hyperparams, #
				 random_seed: int = 0,
				 docker_containers: Dict[str, DockerContainer] = None) -> None:
		super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

		self._clf = MP(window_size=hyperparams['window_size'])

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

	def get_params(self) -> Params:		# pragma: no cover
		"""
		Return parameters.
		Args:
			None

		Returns:
			class Params
		"""
		return super().get_params()

	def set_params(self, *, params: Params) -> None:	# pragma: no cover
		"""
		Set parameters for outlier detection.
		Args:
			params: class Params

		Returns:
			None
		"""
		super().set_params(params=params)
