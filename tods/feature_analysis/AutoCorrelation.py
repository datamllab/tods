import os
import typing
import collections
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import logging, uuid
from scipy import sparse
from numpy import ndarray
from collections import OrderedDict
from common_primitives import dataframe_utils, utils

from d3m.base import utils as base_utils
from d3m.primitive_interfaces import base, transformer
from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams

from statsmodels.tsa.stattools import acf

__all__ = ('AutoCorrelation',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
	"""
	AutoCorrelation = hyperparams.Enumeration(
		values = ["acf", "pacf", "pacf_yw", "pacf_ols"],
		default = "acf",
		semantic_types=[],
		description='AutoCorrelation to use'
	)
	"""
	unbiased = hyperparams.UniformBool(
		default=False,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="If True, then denominators for autocovariance are n-k, otherwise n."
	)
	nlags = hyperparams.UniformInt(
		lower = 0,
		upper = 100,	#TODO: Define the correct the upper bound
		default=40,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Number of lags to return autocorrelation for."
	)
	qstat = hyperparams.UniformBool(
		default=False,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="If True, returns the Ljung-Box q statistic for each autocorrelationcoefficient."
	)
	fft = hyperparams.UniformBool(
		default=False,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="If True, computes the ACF via FFT."
	)
	alpha = hyperparams.Bounded[float](
		lower=0,
		upper=1,
		lower_inclusive=True,
		upper_inclusive=True,
		default = 0,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="""If a number is given, the confidence intervals for the given level are returned.
		For instance if alpha=.05, 95 % confidence intervals are returned where the standard deviation is computed according to Bartlett"s formula."""
	)
	missing = hyperparams.Enumeration[str](
		values=["none", "raise", "conservative", "drop"],
		default="none",
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="""Specifying how the NaNs are to be treated. "none" performs no checks. "raise" raises an exception if NaN values are found. 
		"drop" removes the missing observations and then estimates the autocovariances treating the non-missing as contiguous. 
		"conservative" computes the autocovariance using nan-ops so that nans are removed when computing the mean
		and cross-products that are used to estimate the autocovariance.
		When using "conservative", n is set to the number of non-missing observations."""
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

class ACF:
	"""
	This is the class for autocorrelation function
	"""
	def __init__(self, unbiased=False, nlags=40, qstat=False, fft=None, alpha=None,missing="none"):
		self._unbiased = unbiased
		self._nlags = nlags
		self._qstat = qstat
		self._fft = fft
		self._alpha = alpha
		self._missing = missing

	def produce(self, data):

		"""

		Args:
			data: dataframe column
		Returns:
			nparray

		"""

		output = acf(data)
		return output



class AutoCorrelation(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
	"""
	A primitive that performs autocorrelation on a DataFrame
	acf() function documentation: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.acf.html
	"""

	__author__ = "DATA Lab @Texas A&M University"
	metadata = metadata_base.PrimitiveMetadata(
		{
			'id': '8c246c78-3082-4ec9-844e-5c98fcc76f9f',
			'version': '0.0.2',
			'name': "AutoCorrelation of values",
			'python_path': 'd3m.primitives.tods.feature_analysis.auto_correlation',
			'algorithm_types': [metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,], #TODO: check is this right?
			'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
			"hyperparams_to_tune": ['unbiased', 'nlags', 'qstat', 'fft', 'alpha', 'missing'],
			'source': {
				'name': 'DATA Lab @Texas A&M University',
				'contact': 'mailto:khlai037@tamu.edu',
				'uris': ['https://gitlab.com/lhenry15/tods/-/blob/Yile/anomaly-primitives/anomaly_primitives/AutoCorrelation.py'],
			},
			'installation': [{
				'type': metadata_base.PrimitiveInstallationType.PIP,
				'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
					git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
				),
			}],		
		},
	)


	def __init__(self, *, hyperparams: Hyperparams) -> None:
	 	super().__init__(hyperparams=hyperparams)

	 	self._clf = ACF(unbiased = hyperparams['unbiased'],
						nlags = hyperparams['nlags'],
						qstat = hyperparams['qstat'],
						fft = hyperparams['fft'],
						alpha = hyperparams['alpha'],
						missing = hyperparams['missing']
	 				)

	def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:	
		"""
		Args:
			inputs: Container DataFrame
			timeout: Default
			iterations: Default
		Returns:
		    Container DataFrame containing moving average of selected columns
		"""

		assert isinstance(inputs, container.DataFrame), type(container.DataFrame)
		_, self._columns_to_produce = self._get_columns_to_fit(inputs, self.hyperparams)
		
		
		outputs = inputs
		if len(self._columns_to_produce) > 0:
			for col in self.hyperparams['use_columns']:
				output = self._clf.produce(inputs.iloc[ : ,col])
				outputs = pd.concat((outputs, pd.Series(output).rename(inputs.columns[col] + '_acf')), axis = 1)
		else:
			if self.hyperparams['error_on_no_input']:
				raise RuntimeError("No input columns were selected")
			self.logger.warn("No input columns were selected")

		self._update_metadata(outputs)

		return base.CallResult(outputs)



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

		if not issubclass(column_metadata['structural_type'], accepted_structural_types):
			return False

		semantic_types = set(column_metadata.get('semantic_types', []))

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

		outputs = container.DataFrame(predictions, generate_metadata=True)
		target_columns_metadata = self._copy_inputs_metadata(inputs.metadata, self._columns_to_produce, outputs.metadata, self.hyperparams)
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
	def _copy_inputs_metadata(cls, inputs_metadata: metadata_base.DataMetadata, input_indices: List[int],
							  outputs_metadata: metadata_base.DataMetadata, hyperparams):
		"""
                Updata metadata for selected columns.

                Args:
                        inputs_metadata: metadata.base.DataMetadata
                        input_indices: list
                        outputs_metadata: metadata.base.DataMetadata
                        hyperparams: d3m.metadata.hyperparams.Hyperparams

                Returns:
                        d3m.metadata.base.DataMetadata
		"""

		outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
		target_columns_metadata: List[OrderedDict] = []
		for column_index in input_indices:
			column_name = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
			if column_name is None:
				column_name = "output_{}".format(column_index)

			column_metadata = OrderedDict(inputs_metadata.query_column(column_index))
			semantic_types = set(column_metadata.get('semantic_types', []))
			semantic_types_to_remove = set([])
			add_semantic_types = set()
			add_semantic_types.add(hyperparams["return_semantic_type"])
			semantic_types = semantic_types - semantic_types_to_remove
			semantic_types = semantic_types.union(add_semantic_types)
			column_metadata['semantic_types'] = list(semantic_types)

			column_metadata["name"] = str(column_name)
			target_columns_metadata.append(column_metadata)

		#  If outputs has more columns than index, add Attribute Type to all remaining
		if outputs_length > len(input_indices):
			for column_index in range(len(input_indices), outputs_length):
				column_metadata = OrderedDict()
				semantic_types = set()
				semantic_types.add(hyperparams["return_semantic_type"])
				column_name = "output_{}".format(column_index)
				column_metadata["semantic_types"] = list(semantic_types)
				column_metadata["name"] = str(column_name)
				target_columns_metadata.append(column_metadata)
		return target_columns_metadata
