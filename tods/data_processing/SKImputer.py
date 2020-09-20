from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _get_mask


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
    statistics_: Optional[ndarray]
    indicator_: Optional[sklearn.base.BaseEstimator]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    missing_values = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Hyperparameter[int](
                default=0,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'float': hyperparams.Hyperparameter[float](
                default=numpy.nan,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='float',
        description='The placeholder for the missing values. All occurrences of `missing_values` will be imputed.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    strategy = hyperparams.Enumeration[str](
        default='mean',
        values=['median', 'most_frequent', 'mean', 'constant'],
        description='The imputation strategy.  - If "mean", then replace missing values using the mean along each column. Can only be used with numeric data. - If "median", then replace missing values using the median along each column. Can only be used with numeric data. - If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data. - If "constant", then replace missing values with fill_value. Can be used with strings or numeric data.  .. versionadded:: 0.20 strategy="constant" for fixed value imputation.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    add_indicator = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    fill_value = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Hyperparameter[int](
                default=0,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='When strategy == "constant", fill_value is used to replace all occurrences of missing_values. If left to the default, fill_value will be 0 when imputing numerical data and "missing_value" for strings or object data types.',
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
    
    return_semantic_type = hyperparams.Enumeration[str](
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class SKImputer(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn SimpleImputer
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_
    
    """
    
    __author__ = "DataLab @ Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.IMPUTATION, ],
         "name": "sklearn.impute.SimpleImputer",
         "primitive_family": metadata_base.PrimitiveFamily.DATA_CLEANING,
         "python_path": "d3m.primitives.tods.data_processing.impute_missing",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html']},
         "version": "2019.11.13",
         "id": "d016df89-de62-3c53-87ed-c06bb6a23cde",
         "hyperparams_to_tune": ['strategy'],
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
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = SimpleImputer(
              missing_values=self.hyperparams['missing_values'],
              strategy=self.hyperparams['strategy'],
              add_indicator=self.hyperparams['add_indicator'],
              fill_value=self.hyperparams['fill_value'],
              verbose=_verbose
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

        self._training_inputs, self._training_indices, _ = self._get_columns_to_fit(self._inputs, self.hyperparams)
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
        sk_inputs, columns_to_use, _ = self._get_columns_to_fit(inputs, self.hyperparams)
        output = []
        if len(sk_inputs.columns):
            try:
                sk_output = self._clf.transform(sk_inputs)
            except sklearn.exceptions.NotFittedError as error:
                raise PrimitiveNotFittedError("Primitive not fitted.") from error
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()
            target_columns_metadata = self._copy_columns_metadata(inputs.metadata, self._training_indices, self.hyperparams)
            output = self._wrap_predictions(inputs, sk_output, target_columns_metadata)

            output.columns = [inputs.columns[idx] for idx in range(len(inputs.columns)) if idx in self._training_indices]
            output = [output]
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        _, _, dropped_cols = self._get_columns_to_fit(inputs, self.hyperparams)
        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._training_indices + dropped_cols,
                                               columns_list=output)
        return CallResult(outputs)
        

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                statistics_=None,
                indicator_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            statistics_=getattr(self._clf, 'statistics_', None),
            indicator_=getattr(self._clf, 'indicator_', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.statistics_ = params['statistics_']
        self._clf.indicator_ = params['indicator_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['statistics_'] is not None:
            self._fitted = True
        if params['indicator_'] is not None:
            self._fitted = True



    
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):

        if not hyperparams['use_semantic_types']:
            columns_to_produce = list(range(len(inputs.columns)))

        else:
            inputs_metadata = inputs.metadata

            def can_produce_column(column_index: int) -> bool:
                return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

            columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                                 use_columns=hyperparams['use_columns'],
                                                                                 exclude_columns=hyperparams['exclude_columns'],
                                                                                 can_use_column=can_produce_column)

        columns_to_drop = cls._get_columns_to_drop(inputs, columns_to_produce, hyperparams)
        for col in columns_to_drop:
            columns_to_produce.remove(col)

        return inputs.iloc[:, columns_to_produce], columns_to_produce, columns_to_drop

    @classmethod
    def _get_columns_to_drop(cls, inputs: Inputs, column_indices: List[int], hyperparams: Hyperparams):
        """
        Check for columns that contain missing_values that need to be imputed
        If strategy is constant and missin_values is nan, then all nan columns will not be dropped
        :param inputs:
        :param column_indices:
        :return:
        """
        columns_to_remove = []
        if hyperparams['strategy'] != "constant":
            for _, col in enumerate(column_indices):
                    inp = inputs.iloc[:, [col]].values
                    mask = _get_mask(inp, hyperparams['missing_values'])
                    if mask.all():
                        columns_to_remove.append(col)
        return columns_to_remove

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
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

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray, target_columns_metadata) -> Outputs:
        outputs = d3m_dataframe(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    
    @classmethod
    def _copy_columns_metadata(cls, inputs_metadata: metadata_base.DataMetadata, column_indices, hyperparams) -> List[OrderedDict]:
        outputs_length = inputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in column_indices:
            column_name = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
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

        return target_columns_metadata


SKImputer.__doc__ = SimpleImputer.__doc__
