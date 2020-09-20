from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import time

# Custom import commands if any
from sklearn.decomposition.truncated_svd import TruncatedSVD


from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase


Inputs = d3m_dataframe
Outputs = d3m_dataframe

__all__ = ('SKTruncatedSVD',)

class PrimitiveCount:
    primitive_no = 0

class Params(params.Params):
    components_: Optional[ndarray]
    explained_variance_ratio_: Optional[ndarray]
    explained_variance_: Optional[ndarray]
    singular_values_: Optional[ndarray]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]


class Hyperparams(hyperparams.Hyperparams):
    n_components = hyperparams.Bounded[int](
        default=2,
        lower=0,
        upper=None,
        description='Desired dimensionality of output data. Must be strictly less than the number of features. The default value is useful for visualisation. For LSA, a value of 100 is recommended.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    algorithm = hyperparams.Choice(
        choices={
            'randomized': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'n_iter': hyperparams.Bounded[int](
                        default=5,
                        lower=0,
                        upper=None,
                        description='Number of iterations for randomized SVD solver. Not used in arpack',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'arpack': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'tol': hyperparams.Bounded[float](
                        default=0,
                        lower=0,
                        upper=None,
                        description='Tolerance for ARPACK. 0 means machine precision. Ignored by randomized SVD solver.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            )
        },
        default='randomized',
        description='SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds), or "randomized" for the randomized algorithm due to Halko (2009).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
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
        default='append',
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
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class SKTruncatedSVD(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn TruncatedSVD
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_   

    Parameters
    ----------
    n_components: int
        Desired dimensionality of output data. Must be strictly less than the number of features. The default value is useful for visualisation. For LSA, a value of 100 is recommended.

    algorithm: hyperparams.Choice
       SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy (scipy.sparse.linalg.svds), or "randomized" for the randomized algorithm due to Halko (2009).
    
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

    __author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({
         "name": "Truncated SVD",
         "python_path": "d3m.primitives.tods.feature_analysis.truncated_svd",
         "source": {'name': 'DATA Lab at Texas A&M University', 'contact': 'mailto:khlai037@tamu.edu', 
         'uris': ['https://gitlab.com/lhenry15/tods.git', 'https://gitlab.com/lhenry15/tods/-/blob/Junjie/anomaly-primitives/anomaly_primitives/SKTruncatedSVD.py']},
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.SINGULAR_VALUE_DECOMPOSITION, ],
         "primitive_family": metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
         "id": "9231fde3-7322-3c41-b4cf-d00a93558c44",
         "hyperparams_to_tune": ['n_components', 'algorithm', 'use_columns', 'exclude_columns', 'return_result', 'use_semantic_types', 'add_index_columns', 'error_on_no_input', 'return_semantic_type'],
         "version": "0.0.1",
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = TruncatedSVD(
              n_components=self.hyperparams['n_components'],
              algorithm=self.hyperparams['algorithm']['choice'],
              n_iter=self.hyperparams['algorithm'].get('n_iter', 5),
              tol=self.hyperparams['algorithm'].get('tol', 0),
              random_state=self.random_seed,
        )

        self.primitiveNo = PrimitiveCount.primitive_no
        PrimitiveCount.primitive_no += 1

        
        
        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False
        
        
    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for SKTruncatedSVD.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        # self.logger.warning('set was called!')
        self._inputs = inputs
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        if self._fitted:
            return CallResult(None)

        # Get cols to fit.
        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        # If there is no cols to fit, return None
        if self._training_inputs is None:
            return CallResult(None)

        # Call SVD in sklearn and set _fitted to true
        if len(self._training_indices) > 0:
            self._clf.fit(self._training_inputs)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame.

        Returns:
            Container DataFrame after Truncated SVD.
        """
        # self.logger.warning(str(self.metadata.query()['name']))


        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            sk_output = self._clf.transform(sk_inputs)
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

        # self._write(outputs)
        # self.logger.warning('produce was called!')
        return CallResult(outputs)
        

    def get_params(self) -> Params:
        """
        Return parameters.
        Args:
            None

        Returns:
            class Params
        """
        if not self._fitted:
            return Params(
                components_=None,
                explained_variance_ratio_=None,
                explained_variance_=None,
                singular_values_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            components_=getattr(self._clf, 'components_', None),
            explained_variance_ratio_=getattr(self._clf, 'explained_variance_ratio_', None),
            explained_variance_=getattr(self._clf, 'explained_variance_', None),
            singular_values_=getattr(self._clf, 'singular_values_', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters for SKTruncatedSVD.
        Args:
            params: class Params

        Returns:
            None
        """
        self._clf.components_ = params['components_']
        self._clf.explained_variance_ratio_ = params['explained_variance_ratio_']
        self._clf.explained_variance_ = params['explained_variance_']
        self._clf.singular_values_ = params['singular_values_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['components_'] is not None:
            self._fitted = True
        if params['explained_variance_ratio_'] is not None:
            self._fitted = True
        if params['explained_variance_'] is not None:
            self._fitted = True
        if params['singular_values_'] is not None:
            self._fitted = True

   
    
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
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

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

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
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
        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams, self.primitiveNo)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


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

    def _write(self, inputs:Inputs):
        """
        write inputs to current directory, only for test
        """
        inputs.to_csv(str(time.time())+'.csv')


# SKTruncatedSVD.__doc__ = TruncatedSVD.__doc__
