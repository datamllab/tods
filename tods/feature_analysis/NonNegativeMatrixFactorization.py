from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from typing import cast, Dict, List, Union, Sequence, Optional, Tuple
from collections import OrderedDict
from scipy import sparse

import nimfa
import pandas as pd 
import numpy
from numpy import ndarray
import warnings



__all__ = ('NonNegativeMatrixFactorization',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):

	rank = hyperparams.Hyperparameter[int](
		default=30,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="The factorization rank to achieve. Default is 30.",
	)

	seed = hyperparams.Enumeration(
		values=['nndsvd','random_c','random_vcol','random','fixed'],
		default='random',
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="""Method to seed the computation of a factorization""",
	)

	W = hyperparams.Union(
        configuration=OrderedDict({
            'ndarray': hyperparams.Hyperparameter[ndarray](
                default=numpy.array([]),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Score weight by dimensions. If None, [1,1,...,1] will be used.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
			

	H = hyperparams.Union(
        configuration=OrderedDict({
            'ndarray': hyperparams.Hyperparameter[ndarray](
                default=numpy.array([]),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Score weight by dimensions. If None, [1,1,...,1] will be used.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

	update = hyperparams.Enumeration(
		values=['euclidean','divergence'],
		default='euclidean',
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="""Type of update equations used in factorization. When specifying model parameter update can be assigned to:"			
					'euclidean' for classic Euclidean distance update equations,"
					'divergence' for divergence update equations."

					By default Euclidean update equations are used.""",
	)


	objective = hyperparams.Enumeration(
		values=['fro','div','conn'],
		default='fro',
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="""Type of objective function used in factorization. When specifying model parameter :param:`objective` can be assigned to:

					‘fro’ for standard Frobenius distance cost function,
					‘div’ for divergence of target matrix from NMF estimate cost function (KL),
					‘conn’ for measuring the number of consecutive iterations in which the connectivity matrix has not changed.

					By default the standard Frobenius distance cost function is used.""",
	)

	max_iter = hyperparams.Hyperparameter[int](
		default=30,
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Maximum number of factorization iterations. Note that the number of iterations depends on the speed of method convergence. Default is 30.",
	)

	learning_rate = hyperparams.Union[Union[float, None]](
		configuration=OrderedDict(
			limit=hyperparams.Bounded[float](
				lower=0,
				upper=None,
				default=0.01,
			),
			unlimited=hyperparams.Constant(
				default=None,
				description='If nothing is give as a paramter',
			),
		),
		default='unlimited',
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="Minimal required improvement of the residuals from the previous iteration. They are computed between the target matrix and its MF estimate using the objective function associated to the MF algorithm. Default is None.",
	)


	# parameters for column
	use_columns = hyperparams.Set(
		elements=hyperparams.Hyperparameter[int](-1),
		default=(2,3),
		semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
		description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
	)
	exclude_columns = hyperparams.Set(
		elements=hyperparams.Hyperparameter[int](-1),
		default=(),
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



class NMF:
	def __init__(self, rank,W,H,seed,update,objective,max_iter,learning_rate):
		self._rank = rank
		self._seed = seed
		self._W = W,
		self._H = H,
		self._update = update
		self._objective = objective
		self._max_iter = max_iter
		self._learning_rate = learning_rate
		
	def produce(self, inputs):
		
		warnings.filterwarnings("ignore") # for removing warnings thrown by nimfa
		# for testing
		# a = numpy.array([[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1]])
		# b = numpy.array([[1,0],[1,0],[1,0],[1,0],[1,0]])
		# print(type(a))
		# print(type(self._W[0]))
		nmf = nimfa.Nmf(V = numpy.array(inputs.values),
						seed=self._seed,
						W=self._W[0],
						H=self._H[0],
						rank=self._rank,
						update = self._update,
						objective=self._objective,
						min_residuals=self._learning_rate
						)
		nmf_fit = nmf()
		W = nmf_fit.basis()
		H = nmf_fit.coef()

		column_names = ['row_latent_vector_'+str(i) for i in range(self._rank)]
		W = pd.DataFrame(data = W,columns = column_names)
		# print(type(W))

		#TODO: Column latent vector
		column_names = ['column_latent_vector_'+str(i) for i in range(inputs.shape[1])]
		H = pd.DataFrame(data = H,columns = column_names)

		W.reset_index(drop=True, inplace=True)
		H.reset_index(drop=True, inplace=True)
		result = pd.concat([W, H], axis=1)
		# print(result.head(10))
		return result


class NonNegativeMatrixFactorization(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
	"""
	Calculates Latent factors of a given matrix of timeseries data

	Parameters
	----------
	rank: int
		The factorization rank to achieve. Default is 30.

	update: str
		Type of update equations used in factorization. When specifying model parameter update can be assigned to:"			
					'euclidean' for classic Euclidean distance update equations,"
					'divergence' for divergence update equations."

					By default Euclidean update equations are used.

	objective: str
		Type of objective function used in factorization. When specifying model parameter :param:`objective` can be assigned to:

					‘fro’ for standard Frobenius distance cost function,
					‘div’ for divergence of target matrix from NMF estimate cost function (KL),
					‘conn’ for measuring the number of consecutive iterations in which the connectivity matrix has not changed.

					By default the standard Frobenius distance cost function is used.

	max_iter: int
		Maximum number of factorization iterations. Note that the number of iterations depends on the speed of method convergence. Default is 30.

	learning_rate: float
		Minimal required improvement of the residuals from the previous iteration. They are computed between the target matrix and its MF estimate using the objective function associated to the MF algorithm. Default is None.

	
	use_columns: Set
		A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.

	exclude_columns: Set
		A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.

	return_result: Enumeration
		Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.

	use_semantic_types: Bool
		Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe.

	add_index_columns: Bool
		Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".

	error_on_no_input: Bool(
		Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.

	return_semantic_type: Enumeration[str](
		Decides what semantic type to attach to generated attributes'
	"""



	__author__ = "Data Lab"
	metadata = metadata_base.PrimitiveMetadata(
	{
		'__author__' : "DATA Lab at Texas A&M University",
		'name': "Fast Fourier Transform",
		'python_path': 'd3m.primitives.tods.feature_analysis.non_negative_matrix_factorization',
		'source': {
		'name': 'DATA Lab at Texas A&M University',
		'contact': 'mailto:khlai037@tamu.edu',
		'uris': [
			'https://gitlab.com/lhenry15/tods.git',
			'https://gitlab.com/lhenry15/tods/-/blob/purav/anomaly-primitives/anomaly_primitives/NonNegativeMatrixFactorization.py',
		],
		},
		'algorithm_types': [
			metadata_base.PrimitiveAlgorithmType.NON_NEGATIVE_MATRIX_FACTORIZATION,
		],
		'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
		'id': 'c7259da6-7ce6-42ad-83c6-15238679f5fa',
		'hyperparameters_to_tune':['rank','update','objective','max_iter','learning_rate'],
		'version': '0.0.1',
	},
	)

	def __init__(self, *, hyperparams: Hyperparams) -> None:
		super().__init__(hyperparams=hyperparams)



		self._clf = NMF(rank=self.hyperparams['rank'],
						seed=self.hyperparams['seed'],
						W=self.hyperparams['W'],
						H=self.hyperparams['H'],
						objective=self.hyperparams['objective'],
						update=self.hyperparams['update'],
						max_iter=self.hyperparams['max_iter'],
						learning_rate = self.hyperparams['learning_rate'],
						)

	def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

		assert isinstance(inputs, container.DataFrame), type(dataframe)

		self._fitted = False
		self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
		self._input_column_names = self._training_inputs.columns
		
		if len(self._training_indices) > 0:
			# self._clf.fit(self._training_inputs)
			self._fitted = True
		else:
			if self.hyperparams['error_on_no_input']:
				raise RuntimeError("No input columns were selected")
			self.logger.warn("No input columns were selected")
		
		
		if not self._fitted:
			raise PrimitiveNotFittedError("Primitive not fitted.")

		sk_inputs = inputs
		if self.hyperparams['use_semantic_types']:
			cols = [inputs.columns[x] for x in self._training_indices]
			sk_inputs = container.DataFrame(data = inputs.iloc[:, self._training_indices].values,columns = cols, generate_metadata=True)

		output_columns = []
		if len(self._training_indices) > 0:
			sk_output = self._clf.produce(sk_inputs)
			
			if sparse.issparse(sk_output):
				sk_output = sk_output.toarray()
			outputs = self._wrap_predictions(inputs, sk_output)
			# if len(outputs.columns) == len(self._input_column_names):
			#     outputs.columns = self._input_column_names
			output_columns = [outputs]
		else:
			if self.hyperparams['error_on_no_input']:
				raise RuntimeError("No input columns were selected")
			self.logger.warn("No input columns were selected")

		
		outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
											add_index_columns=self.hyperparams['add_index_columns'],
											inputs=inputs, column_indices=self._training_indices,
											columns_list=output_columns)

		
		return base.CallResult(outputs)


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
			# return inputs, list(hyperparams['use_columns'])

		inputs_metadata = inputs.metadata

		def can_produce_column(column_index: int) -> bool:
			return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

		columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
																					use_columns=hyperparams['use_columns'],
																					exclude_columns=hyperparams['exclude_columns'],
																					can_use_column=can_produce_column)
		return inputs.iloc[:, columns_to_produce], columns_to_produce
		# return columns_to_produce


	@classmethod
	def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int,
							hyperparams: Hyperparams) -> bool:
		"""
		Output whether a column can be processed.
		Args:
			inputs_metadata: d3m.metadata.base.DataMetadata
			column_index: int

		Returns:
			bool
		"""

		column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))
		accepted_structural_types = (int, float, numpy.integer, numpy.float64,str)
		accepted_semantic_types = set()
		accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
		if not issubclass(column_metadata['structural_type'], accepted_structural_types):
			print(column_index, "does not match the structural_type requirements in metadata. Skipping column")
			return False

		semantic_types = set(column_metadata.get('semantic_types', []))

		# print("length sematic type",len(semantic_types))

		# returing true for testing purposes for custom dataframes
		return True;

		if len(semantic_types) == 0:
			cls.logger.warning("No semantic types found in column metadata")
			return False

		# Making sure all accepted_semantic_types are available in semantic_types
		if len(accepted_semantic_types - semantic_types) == 0:
			return True

		print(semantic_types)
		return False


	@classmethod
	def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams) -> List[OrderedDict]:
		"""
		Output metadata of selected columns.
		Args:
			outputs_metadata: metadata_base.DataMetadata
			hyperparams: d3m.metadata.hyperparams.Hyperparams

		Returns:
			d3m.metadata.base.DataMetadata
		"""

		outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

		target_columns_metadata: List[OrderedDict] = []
		for column_index in range(outputs_length):
			column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

			# Update semantic types and prepare it for predicted targets.
			semantic_types = set(column_metadata.get('semantic_types', []))
			semantic_types_to_remove = set([])
			add_semantic_types = []
			add_semantic_types.add(hyperparams["return_semantic_type"])
			semantic_types = semantic_types - semantic_types_to_remove
			semantic_types = semantic_types.union(add_semantic_types)
			column_metadata['semantic_types'] = list(semantic_types)

			target_columns_metadata.append(column_metadata)

		return target_columns_metadata


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
		target_columns_metadata = self._add_target_columns_metadata(outputs.metadata,self.hyperparams)
		outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
		# print(outputs.metadata.to_internal_simple_structure())

		return outputs


	@classmethod
	def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams):
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
			# column_name = "output_{}".format(column_index)
			column_metadata = OrderedDict()
			semantic_types = set()
			semantic_types.add(hyperparams["return_semantic_type"])
			column_metadata['semantic_types'] = list(semantic_types)
			# column_metadata["name"] = str(column_name)
			target_columns_metadata.append(column_metadata)
		return target_columns_metadata

NonNegativeMatrixFactorization.__doc__ = NonNegativeMatrixFactorization.__doc__
