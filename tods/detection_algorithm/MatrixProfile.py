import os
import sklearn
import numpy
import typing
import time
from scipy import sparse
from numpy import ndarray
from collections import OrderedDict
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import logging, uuid
from scipy import sparse
from numpy import ndarray
from collections import OrderedDict
from common_primitives import dataframe_utils, utils

from d3m import utils
from d3m import container
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.container import DataFrame as d3m_dataframe
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer

import stumpy

__all__ = ('MatrixProfile',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class PrimitiveCount:
    primitive_no = 0


class Hyperparams(hyperparams.Hyperparams):
	window_size = hyperparams.UniformInt(
		lower = 0,
		upper = 100,	#TODO: Define the correct the upper bound
		default=50,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="window size to calculate"
	)
	
	# Keep previous
	dataframe_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
		default=None,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Resource ID of a DataFrame to extract if there are multiple tabular resources inside a Dataset and none is a dataset entry point.",
	)
	use_columns = hyperparams.Set(
		elements=hyperparams.Hyperparameter[int](-1),
		default=(2,),
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
	)
	exclude_columns = hyperparams.Set(
		elements=hyperparams.Hyperparameter[int](-1),
		default=(0,1,3,),
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
	)
	return_result = hyperparams.Enumeration(
		values=['append', 'replace', 'new'],
		default='new',
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
	)
	use_semantic_types = hyperparams.UniformBool(
		default=False,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
	)
	add_index_columns = hyperparams.UniformBool(
		default=False,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
	)
	error_on_no_input = hyperparams.UniformBool(
		default=True,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.",
	)
	return_semantic_type = hyperparams.Enumeration[str](
		values=['https://metadata.datadrivendiscovery.org/types/Attribute',
			'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
		default='https://metadata.datadrivendiscovery.org/types/Attribute',
		description='Decides what semantic type to attach to generated attributes',
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
	)


class MP:
	"""
	This is the class for matrix profile function
	"""
	def __init__(self, window_size):
		self._window_size = window_size
		return

	def produce(self, data):

		"""

		Args:
			data: dataframe column
		Returns:
			nparray

		"""
		transformed_columns=utils.pandas.DataFrame()
		#transformed_columns=d3m_dataframe
		for col in data.columns:
			output = stumpy.stump(data[col], m = self._window_size)
			output = pd.DataFrame(output)
			#print("output", output)
			transformed_columns=pd.concat([transformed_columns,output],axis=1)
			#transformed_columns[col]=output
			#print(transformed_columns)
		return transformed_columns

class MatrixProfile(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
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
    	Returns
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


	def __init__(self, *, hyperparams: Hyperparams) -> None:
		super().__init__(hyperparams=hyperparams)
		self._clf = MP(window_size = hyperparams['window_size'])
		self.primitiveNo = PrimitiveCount.primitive_no
		PrimitiveCount.primitive_no+=1

	def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:	

		"""

		Args:

			inputs: Container DataFrame

			timeout: Default

			iterations: Default

		Returns:

		    Container DataFrame containing Matrix Profile of selected columns
		
		"""

		# Get cols to fit.
		self._fitted = False
		self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
		self._input_column_names = self._training_inputs.columns


		if len(self._training_indices) > 0:
			self._fitted = True
		else:
			if self.hyperparams['error_on_no_input']:
				raise RuntimeError("No input columns were selected")
			self.logger.warn("No input columns were selected")

		if not self._fitted:
			raise PrimitiveNotFittedError("Primitive not fitted.")
		
		sk_inputs = inputs
		if self.hyperparams['use_semantic_types']:
			sk_inputs = inputs.iloc[:, self._training_indices]
		output_columns = []
		if len(self._training_indices) > 0:
			sk_output = self._clf.produce(sk_inputs)
			if sparse.issparse(sk_output):
				sk_output = sk_output.toarray()
			outputs = self._wrap_predictions(inputs, sk_output)
			
			if len(outputs.columns) == len(self._input_column_names):
				outputs.columns = self._input_column_names
			output_columns = [outputs]

		else:
			if self.hyperparams['error_on_no_input']:
				raise RuntimeError("No input columns were selected")
			self.logger.warn("No input columns were selected")

		outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
							   add_index_columns=self.hyperparams['add_index_columns'],
							   inputs=inputs, column_indices=self._training_indices,
							   columns_list=output_columns)
		#print(outputs)
		#CallResult(outputs)
		#print("___")
		print(outputs.columns)
		#outputs.columns = [str(x) for x in outputs.columns]

		return CallResult(outputs)

		# assert isinstance(inputs, container.DataFrame), type(container.DataFrame)
		# _, self._columns_to_produce = self._get_columns_to_fit(inputs, self.hyperparams)
		
		# #print("columns_to_produce ", self._columns_to_produce)
		
		# outputs = inputs
		# if len(self._columns_to_produce) > 0:
		# 	for col in self.hyperparams['use_columns']:
		# 		output = self._clf.produce(inputs.iloc[ : ,col])
				
		# 		outputs = pd.concat((outputs, pd.DataFrame({inputs.columns[col]+'_matrix_profile': output[:,0], 
		# 					inputs.columns[col]+'_matrix_profile_indices': output[:,1], 
		# 					inputs.columns[col]+'_left_matrix_profile_indices': output[:,2], 
		# 					inputs.columns[col]+'_right_matrix_profile_indices': output[:,3]})), axis = 1)

		# else:
		# 	if self.hyperparams['error_on_no_input']:
		# 		raise RuntimeError("No input columns were selected")
		# 	self.logger.warn("No input columns were selected")

		# #print(outputs)
		# self._update_metadata(outputs)

		# return base.CallResult(outputs)



	def _update_metadata(self, outputs):
		outputs.metadata = outputs.metadata.generate(outputs)
 
	@classmethod
	def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):

		"""

			Select columns to fit.
			Args:
				inputs: Container DataFrame
				hyperparams: d3m.metadata.hyperparams.Hyperparams

			Returns:
				list

		"""

		if not hyperparams['use_semantic_types']:
			return inputs, list(range(len(inputs.columns)))

		inputs_metadata = inputs.metadata

		

		def can_produce_column(column_index: int) -> bool:
			return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

		columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
					   use_columns=hyperparams['use_columns'],
					   exclude_columns=hyperparams['exclude_columns'],
					   can_use_column=can_produce_column)


		"""
		Encountered error: when hyperparams['use_columns'] = (2,3) and hyperparams['exclude_columns'] is (1,2)
		columns_to_produce is still [2]
		"""
		return inputs.iloc[:, columns_to_produce], columns_to_produce
		

	@classmethod
	def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:

		"""

			Output whether a column can be processed.
				Args:
					inputs_metadata: d3m.metadata.base.DataMetadata
					column_index: int

				Returns:
					bool

		"""

		column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

		accepted_structural_types = (int, float, np.integer, np.float64) #changed numpy to np
		accepted_semantic_types = set()
		accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")

		# print(column_metadata)
		# print(column_metadata['structural_type'], accepted_structural_types)

		if not issubclass(column_metadata['structural_type'], accepted_structural_types):
			return False

		semantic_types = set(column_metadata.get('semantic_types', []))

		# print(column_metadata)
		# print(semantic_types, accepted_semantic_types)

		if len(semantic_types) == 0:
			cls.logger.warning("No semantic types found in column metadata")
			return False

		# Making sure all accepted_semantic_types are available in semantic_types
		if len(accepted_semantic_types - semantic_types) == 0:
			return True

		return False

	def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:

		"""

			Wrap predictions into dataframe
		Args:
			inputs: Container Dataframe
			predictions: array-like data (n_samples, n_features)

		Returns:
			Dataframe

		"""

		outputs = d3m_dataframe(predictions, generate_metadata=True)
		target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams, self.primitiveNo)
		outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
		return outputs



	@classmethod
	def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
									target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:

		"""

			Updata metadata for selected columns.
				Args:
					inputs_metadata: metadata_base.DataMetadata
					outputs: Container Dataframe
					target_columns_metadata: list

				Returns:
					d3m.metadata.base.DataMetadata

		"""

		outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

		for column_index, column_metadata in enumerate(target_columns_metadata):
			column_metadata.pop("structural_type", None)
			outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

		return outputs_metadata


	@classmethod
	def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams, primitiveNo):
		"""
		Add target columns metadata
		Args:
			outputs_metadata: metadata.base.DataMetadata
			hyperparams: d3m.metadata.hyperparams.Hyperparams

		Returns:
			List[OrderedDict]
		"""
		outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
		target_columns_metadata: List[OrderedDict] = []
		for column_index in range(outputs_length):
			column_name = "{0}{1}_{2}".format(cls.metadata.query()['name'], primitiveNo, column_index)
			column_metadata = OrderedDict()
			semantic_types = set()
			semantic_types.add(hyperparams["return_semantic_type"])
			column_metadata['semantic_types'] = list(semantic_types)

			column_metadata["name"] = str(column_name)
			target_columns_metadata.append(column_metadata)
		return target_columns_metadata
