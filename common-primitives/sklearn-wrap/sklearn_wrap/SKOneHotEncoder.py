from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.preprocessing.data import OneHotEncoder
from numpy import float as npfloat


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


class Params(params.Params):
    _active_features_: Optional[ndarray]
    _categorical_features: Optional[Union[str, ndarray]]
    _categories: Optional[Sequence[Any]]
    _feature_indices_: Optional[ndarray]
    _legacy_mode: Optional[bool]
    _n_values_: Optional[ndarray]
    _n_values: Optional[Union[str, ndarray]]
    categories_: Optional[Sequence[Any]]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    n_values = hyperparams.Union(
        configuration=OrderedDict({
            'auto': hyperparams.Constant(
                default='auto',
                description='Determine value range from training data.',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'int': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                default=10,
                description='Number of categorical values per feature. Each feature value should be in range(n_values).',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'list': hyperparams.List(
                default=[],
                elements=hyperparams.Hyperparameter[int](1),
                description='n_values[i] is the number of categorical values in X[:, i]. Each feature value should be in range(n_values[i]).',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='auto',
        description='Number of values per feature.  - \'auto\' : determine value range from training data. - int : number of categorical values per feature. Each feature value should be in ``range(n_values)`` - array : ``n_values[i]`` is the number of categorical values in ``X[:, i]``. Each feature value should be in ``range(n_values[i])``',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    sparse = hyperparams.UniformBool(
        default=True,
        description='Will return sparse matrix if set True else will return an array.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    handle_unknown = hyperparams.Enumeration[str](
        values=['error', 'ignore'],
        default='error',
        description='Whether to raise an error or ignore if a unknown categorical feature is present during transform.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    categories = hyperparams.Constant(
        default='auto',
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
    encode_target_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should it encode also target columns?",
    )
    return_semantic_type = hyperparams.Enumeration[str](
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class SKOneHotEncoder(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn OneHotEncoder
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.ENCODE_ONE_HOT, ],
         "name": "sklearn.preprocessing.data.OneHotEncoder",
         "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
         "python_path": "d3m.primitives.data_transformation.one_hot_encoder.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html']},
         "version": "2019.11.13",
         "id": "c977e879-1bf5-3829-b5b0-39b00233aff5",
         'installation': [
                        {'type': metadata_base.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                            ),
                           }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = OneHotEncoder(
              n_values=self.hyperparams['n_values'],
              sparse=self.hyperparams['sparse'],
              handle_unknown=self.hyperparams['handle_unknown'],
              categories=self.hyperparams['categories'],
        )
        
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
        self._inputs = inputs
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        if self._training_inputs is None:
            return CallResult(None)

        if len(self._training_indices) > 0:
            self._clf.fit(self._training_inputs)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
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
        return CallResult(outputs)
        

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                _active_features_=None,
                _categorical_features=None,
                _categories=None,
                _feature_indices_=None,
                _legacy_mode=None,
                _n_values_=None,
                _n_values=None,
                categories_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            _active_features_=getattr(self._clf, '_active_features_', None),
            _categorical_features=getattr(self._clf, '_categorical_features', None),
            _categories=getattr(self._clf, '_categories', None),
            _feature_indices_=getattr(self._clf, '_feature_indices_', None),
            _legacy_mode=getattr(self._clf, '_legacy_mode', None),
            _n_values_=getattr(self._clf, '_n_values_', None),
            _n_values=getattr(self._clf, '_n_values', None),
            categories_=getattr(self._clf, 'categories_', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        self._clf._active_features_ = params['_active_features_']
        self._clf._categorical_features = params['_categorical_features']
        self._clf._categories = params['_categories']
        self._clf._feature_indices_ = params['_feature_indices_']
        self._clf._legacy_mode = params['_legacy_mode']
        self._clf._n_values_ = params['_n_values_']
        self._clf._n_values = params['_n_values']
        self._clf.categories_ = params['categories_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['_active_features_'] is not None:
            self._fitted = True
        if params['_categorical_features'] is not None:
            self._fitted = True
        if params['_categories'] is not None:
            self._fitted = True
        if params['_feature_indices_'] is not None:
            self._fitted = True
        if params['_legacy_mode'] is not None:
            self._fitted = True
        if params['_n_values_'] is not None:
            self._fitted = True
        if params['_n_values'] is not None:
            self._fitted = True
        if params['categories_'] is not None:
            self._fitted = True



    
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
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
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int,float,numpy.integer,numpy.float64,str,)
        accepted_semantic_types = set(["https://metadata.datadrivendiscovery.org/types/CategoricalData","https://metadata.datadrivendiscovery.org/types/Attribute",])
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        
        if hyperparams['encode_target_columns'] and 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
            return True
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False
    

    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams) -> List[OrderedDict]:
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
        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._copy_inputs_metadata(inputs.metadata, self._training_indices, outputs.metadata, self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    @classmethod
    def _copy_inputs_metadata(cls, inputs_metadata: metadata_base.DataMetadata, input_indices: List[int],
                                        outputs_metadata: metadata_base.DataMetadata, hyperparams):
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


SKOneHotEncoder.__doc__ = OneHotEncoder.__doc__