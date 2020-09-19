from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.svm.classes import SVR


from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas



Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    support_: Optional[ndarray]
    support_vectors_: Optional[ndarray]
    dual_coef_: Optional[ndarray]
    intercept_: Optional[ndarray]
    _sparse: Optional[bool]
    shape_fit_: Optional[tuple]
    n_support_: Optional[ndarray]
    probA_: Optional[ndarray]
    probB_: Optional[ndarray]
    _gamma: Optional[float]
    _dual_coef_: Optional[ndarray]
    _intercept_: Optional[ndarray]
    class_weight_: Optional[ndarray]
    fit_status_: Optional[int]
    class_weight: Optional[Union[str, Dict, List[Dict]]]
    nu: Optional[float]
    probability: Optional[bool]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    C = hyperparams.Bounded[float](
        default=1,
        lower=0,
        upper=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Penalty parameter C of the error term.'
    )
    epsilon = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0.1,
        description='Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    kernel = hyperparams.Choice(
        choices={
            'linear': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            ),
            'poly': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'degree': hyperparams.Bounded[int](
                        default=3,
                        lower=0,
                        upper=None,
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'gamma': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Bounded[float](
                                default=0.1,
                                lower=0,
                                upper=None,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            'auto': hyperparams.Constant(
                                default='auto',
                                description='1/n_features will be used.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            )
                        }),
                        default='auto',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'coef0': hyperparams.Constant(
                        default=0,
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'rbf': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'gamma': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Bounded[float](
                                default=0.1,
                                lower=0,
                                upper=None,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            'auto': hyperparams.Constant(
                                default='auto',
                                description='1/n_features will be used.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            )
                        }),
                        default='auto',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'sigmoid': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'gamma': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Bounded[float](
                                default=0.1,
                                lower=0,
                                upper=None,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            'auto': hyperparams.Constant(
                                default='auto',
                                description='1/n_features will be used.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            )
                        }),
                        default='auto',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'coef0': hyperparams.Constant(
                        default=0,
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'precomputed': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            )
        },
        default='rbf',
        description='Specifies the kernel type to be used in the algorithm. It must be one of \'linear\', \'poly\', \'rbf\', \'sigmoid\', \'precomputed\' or a callable. If none is given, \'rbf\' will be used. If a callable is given it is used to precompute the kernel matrix.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    shrinking = hyperparams.UniformBool(
        default=True,
        description='Whether to use the shrinking heuristic.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    tol = hyperparams.Bounded[float](
        default=0.001,
        lower=0,
        upper=None,
        description='Tolerance for stopping criterion.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    cache_size = hyperparams.Bounded[float](
        default=200,
        lower=0,
        upper=None,
        description='Specify the size of the kernel cache (in MB).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter']
    )
    max_iter = hyperparams.Bounded[int](
        default=-1,
        lower=-1,
        upper=None,
        description='Hard limit on iterations within solver, or -1 for no limit.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    
    use_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training input. If any specified column cannot be parsed, it is skipped.",
    )
    use_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training target. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training inputs. Applicable only if \"use_columns\" is not provided.",
    )
    exclude_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training target. Applicable only if \"use_columns\" is not provided.",
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
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
        default='https://metadata.datadrivendiscovery.org/types/PredictedTarget',
        description='Decides what semantic type to attach to generated output',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class SKSVR(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn SVR
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.SUPPORT_VECTOR_MACHINE, ],
         "name": "sklearn.svm.classes.SVR",
         "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
         "python_path": "d3m.primitives.regression.svr.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html']},
         "version": "2019.11.13",
         "id": "ebbc3404-902d-33cc-a10c-e42b06dfe60c",
         "hyperparams_to_tune": ['C', 'kernel'],
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
        self._clf = SVR(
              C=self.hyperparams['C'],
              epsilon=self.hyperparams['epsilon'],
              kernel=self.hyperparams['kernel']['choice'],
              degree=self.hyperparams['kernel'].get('degree', 3),
              gamma=self.hyperparams['kernel'].get('gamma', 'auto'),
              coef0=self.hyperparams['kernel'].get('coef0', 0),
              shrinking=self.hyperparams['shrinking'],
              tol=self.hyperparams['tol'],
              cache_size=self.hyperparams['cache_size'],
              max_iter=self.hyperparams['max_iter'],
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
        self._new_training_data = False
        
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._fitted = False
        self._new_training_data = True
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._inputs is None or self._outputs is None:
            raise ValueError("Missing training data.")

        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._training_outputs, self._target_names, self._target_column_indices = self._get_targets(self._outputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        if len(self._training_indices) > 0 and len(self._target_column_indices) > 0:
            self._target_columns_metadata = self._get_target_columns_metadata(self._training_outputs.metadata, self.hyperparams)
            sk_training_output = self._training_outputs.values

            shape = sk_training_output.shape
            if len(shape) == 2 and shape[1] == 1:
                sk_training_output = numpy.ravel(sk_training_output)

            self._clf.fit(self._training_inputs, sk_training_output)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        return CallResult(None)

    
    
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        sk_inputs, columns_to_use = self._get_columns_to_fit(inputs, self.hyperparams)
        output = []
        if len(sk_inputs.columns):
            try:
                sk_output = self._clf.predict(sk_inputs)
            except sklearn.exceptions.NotFittedError as error:
                raise PrimitiveNotFittedError("Primitive not fitted.") from error
            # For primitives that allow predicting without fitting like GaussianProcessRegressor
            if not self._fitted:
                raise PrimitiveNotFittedError("Primitive not fitted.")
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()
            output = self._wrap_predictions(inputs, sk_output)
            output.columns = self._target_names
            output = [output]
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._target_column_indices,
                                               columns_list=output)

        return CallResult(outputs)
        

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                support_=None,
                support_vectors_=None,
                dual_coef_=None,
                intercept_=None,
                _sparse=None,
                shape_fit_=None,
                n_support_=None,
                probA_=None,
                probB_=None,
                _gamma=None,
                _dual_coef_=None,
                _intercept_=None,
                class_weight_=None,
                fit_status_=None,
                class_weight=None,
                nu=None,
                probability=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            support_=getattr(self._clf, 'support_', None),
            support_vectors_=getattr(self._clf, 'support_vectors_', None),
            dual_coef_=getattr(self._clf, 'dual_coef_', None),
            intercept_=getattr(self._clf, 'intercept_', None),
            _sparse=getattr(self._clf, '_sparse', None),
            shape_fit_=getattr(self._clf, 'shape_fit_', None),
            n_support_=getattr(self._clf, 'n_support_', None),
            probA_=getattr(self._clf, 'probA_', None),
            probB_=getattr(self._clf, 'probB_', None),
            _gamma=getattr(self._clf, '_gamma', None),
            _dual_coef_=getattr(self._clf, '_dual_coef_', None),
            _intercept_=getattr(self._clf, '_intercept_', None),
            class_weight_=getattr(self._clf, 'class_weight_', None),
            fit_status_=getattr(self._clf, 'fit_status_', None),
            class_weight=getattr(self._clf, 'class_weight', None),
            nu=getattr(self._clf, 'nu', None),
            probability=getattr(self._clf, 'probability', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.support_ = params['support_']
        self._clf.support_vectors_ = params['support_vectors_']
        self._clf.dual_coef_ = params['dual_coef_']
        self._clf.intercept_ = params['intercept_']
        self._clf._sparse = params['_sparse']
        self._clf.shape_fit_ = params['shape_fit_']
        self._clf.n_support_ = params['n_support_']
        self._clf.probA_ = params['probA_']
        self._clf.probB_ = params['probB_']
        self._clf._gamma = params['_gamma']
        self._clf._dual_coef_ = params['_dual_coef_']
        self._clf._intercept_ = params['_intercept_']
        self._clf.class_weight_ = params['class_weight_']
        self._clf.fit_status_ = params['fit_status_']
        self._clf.class_weight = params['class_weight']
        self._clf.nu = params['nu']
        self._clf.probability = params['probability']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['support_'] is not None:
            self._fitted = True
        if params['support_vectors_'] is not None:
            self._fitted = True
        if params['dual_coef_'] is not None:
            self._fitted = True
        if params['intercept_'] is not None:
            self._fitted = True
        if params['_sparse'] is not None:
            self._fitted = True
        if params['shape_fit_'] is not None:
            self._fitted = True
        if params['n_support_'] is not None:
            self._fitted = True
        if params['probA_'] is not None:
            self._fitted = True
        if params['probB_'] is not None:
            self._fitted = True
        if params['_gamma'] is not None:
            self._fitted = True
        if params['_dual_coef_'] is not None:
            self._fitted = True
        if params['_intercept_'] is not None:
            self._fitted = True
        if params['class_weight_'] is not None:
            self._fitted = True
        if params['fit_status_'] is not None:
            self._fitted = True
        if params['class_weight'] is not None:
            self._fitted = True
        if params['nu'] is not None:
            self._fitted = True
        if params['probability'] is not None:
            self._fitted = True


    


    
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_inputs_columns'],
                                                                             exclude_columns=hyperparams['exclude_inputs_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

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
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return data, list(data.columns), list(range(len(data.columns)))

        metadata = data.metadata

        def can_produce_column(column_index: int) -> bool:
            accepted_semantic_types = set()
            accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/TrueTarget")
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = set(column_metadata.get('semantic_types', []))
            if len(semantic_types) == 0:
                cls.logger.warning("No semantic types found in column metadata")
                return False
            # Making sure all accepted_semantic_types are available in semantic_types
            if len(accepted_semantic_types - semantic_types) == 0:
                return True
            return False

        target_column_indices, target_columns_not_to_produce = base_utils.get_columns_to_use(metadata,
                                                                                               use_columns=hyperparams[
                                                                                                   'use_outputs_columns'],
                                                                                               exclude_columns=
                                                                                               hyperparams[
                                                                                                   'exclude_outputs_columns'],
                                                                                               can_use_column=can_produce_column)
        targets = []
        if target_column_indices:
            targets = data.select_columns(target_column_indices)
        target_column_names = []
        for idx in target_column_indices:
            target_column_names.append(data.columns[idx])
        return targets, target_column_names, target_column_indices

    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams) -> List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = set(column_metadata.get('semantic_types', []))
            semantic_types_to_remove = set(["https://metadata.datadrivendiscovery.org/types/TrueTarget","https://metadata.datadrivendiscovery.org/types/SuggestedTarget",])
            add_semantic_types = set(["https://metadata.datadrivendiscovery.org/types/PredictedTarget",])
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
        outputs = d3m_dataframe(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, self._target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata):
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict()
            semantic_types = []
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            column_name = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)
            column_metadata["semantic_types"] = semantic_types
            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


SKSVR.__doc__ = SVR.__doc__