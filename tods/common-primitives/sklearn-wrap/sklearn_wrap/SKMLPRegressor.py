from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.neural_network.multilayer_perceptron import MLPRegressor


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
    loss_: Optional[float]
    coefs_: Optional[Sequence[Any]]
    intercepts_: Optional[Sequence[Any]]
    n_iter_: Optional[int]
    n_layers_: Optional[int]
    n_outputs_: Optional[int]
    out_activation_: Optional[str]
    _best_coefs: Optional[Sequence[Any]]
    _best_intercepts: Optional[Sequence[Any]]
    _no_improvement_count: Optional[int]
    _random_state: Optional[numpy.random.mtrand.RandomState]
    best_validation_score_: Optional[numpy.float64]
    loss_curve_: Optional[Sequence[Any]]
    t_: Optional[int]
    _optimizer: Optional[sklearn.neural_network._stochastic_optimizers.AdamOptimizer]
    validation_scores_: Optional[Sequence[Any]]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    hidden_layer_sizes = hyperparams.List(
        elements=hyperparams.Bounded(1, None, 100),
        default=(100, ),
        min_size=1,
        max_size=None,
        description='The ith element represents the number of neurons in the ith hidden layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    activation = hyperparams.Enumeration[str](
        values=['identity', 'logistic', 'tanh', 'relu'],
        default='relu',
        description='Activation function for the hidden layer.  - \'identity\', no-op activation, useful to implement linear bottleneck, returns f(x) = x  - \'logistic\', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).  - \'tanh\', the hyperbolic tan function, returns f(x) = tanh(x).  - \'relu\', the rectified linear unit function, returns f(x) = max(0, x)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    solver = hyperparams.Choice(
        choices={
            'lbfgs': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            ),
            'sgd': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'learning_rate': hyperparams.Enumeration[str](
                        values=['constant', 'invscaling', 'adaptive'],
                        default='constant',
                        description='Learning rate schedule for weight updates. Only used when solver=’sgd’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'learning_rate_init': hyperparams.Bounded[float](
                        lower=0,
                        upper=None,
                        default=0.001,
                        description='The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'power_t': hyperparams.Bounded[float](
                        lower=0,
                        upper=None,
                        default=0.5,
                        description='The exponent for inverse scaling learning rate. Only used when solver=’sgd’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'shuffle': hyperparams.UniformBool(
                        default=True,
                        description='Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'momentum': hyperparams.Bounded[float](
                        default=0.9,
                        lower=0,
                        upper=1,
                        description='Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'nesterovs_momentum': hyperparams.UniformBool(
                        default=True,
                        description='Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'early_stopping': hyperparams.UniformBool(
                        default=False,
                        description='Whether to use early stopping to terminate training when validation score is not improving.If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'n_iter_no_change': hyperparams.Bounded[int](
                        default=10,
                        lower=1,
                        upper=None,
                        description='Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'adam': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'learning_rate_init': hyperparams.Bounded[float](
                        lower=0,
                        upper=None,
                        default=0.001,
                        description='The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'shuffle': hyperparams.UniformBool(
                        default=True,
                        description='Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'early_stopping': hyperparams.UniformBool(
                        default=False,
                        description='Whether to use early stopping to terminate training when validation score is not improving.If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'beta_1': hyperparams.Bounded[float](
                        default=0.9,
                        lower=0,
                        upper=1,
                        description='Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1).',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'beta_2': hyperparams.Bounded[float](
                        default=0.999,
                        lower=0,
                        upper=1,
                        description='Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1).',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'epsilon': hyperparams.Bounded[float](
                        default=1e-08,
                        lower=0,
                        upper=None,
                        description='Value for numerical stability in adam. Only used when solver=’adam’',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'n_iter_no_change': hyperparams.Bounded[int](
                        default=10,
                        lower=1,
                        upper=None,
                        description='Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’.',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            )
        },
        default='adam',
        description='The solver for weight optimization.  - \'lbfgs\' is an optimizer in the family of quasi-Newton methods.  - \'sgd\' refers to stochastic gradient descent.  - \'adam\' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba  Note: The default solver \'adam\' works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, \'lbfgs\' can converge faster and perform better.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    alpha = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0.0001,
        description='L2 penalty (regularization term) parameter.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    batch_size = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                default=16,
                description='Size of minibatches for stochastic optimizers. If the solver is \'lbfgs\', the classifier will not use minibatch',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'auto': hyperparams.Constant(
                default='auto',
                description='When set to \'auto\', batch_size=min(200, n_samples)',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='auto',
        description='Size of minibatches for stochastic optimizers. If the solver is \'lbfgs\', the classifier will not use minibatch. When set to "auto", `batch_size=min(200, n_samples)`',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_iter = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=200,
        description='Maximum number of iterations. The solver iterates until convergence (determined by \'tol\') or this number of iterations. For stochastic solvers (\'sgd\', \'adam\'), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    tol = hyperparams.Bounded[float](
        default=0.0001,
        lower=0,
        upper=None,
        description='Tolerance for the optimization. When the loss or score is not improving by at least ``tol`` for ``n_iter_no_change`` consecutive iterations, unless ``learning_rate`` is set to \'adaptive\', convergence is considered to be reached and training stops.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    warm_start = hyperparams.UniformBool(
        default=False,
        description='When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. See :term:`the Glossary <warm_start>`.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    validation_fraction = hyperparams.Bounded[float](
        default=0.1,
        lower=0,
        upper=None,
        description='The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True',
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

class SKMLPRegressor(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn MLPRegressor
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.MULTILAYER_PERCEPTRON, ],
         "name": "sklearn.neural_network.multilayer_perceptron.MLPRegressor",
         "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
         "python_path": "d3m.primitives.regression.mlp.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html']},
         "version": "2019.11.13",
         "id": "a4fedbf8-f69a-3440-9423-559291dfbd61",
         "hyperparams_to_tune": ['hidden_layer_sizes', 'activation', 'solver', 'alpha'],
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
                 _verbose: bool = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = MLPRegressor(
              hidden_layer_sizes=self.hyperparams['hidden_layer_sizes'],
              activation=self.hyperparams['activation'],
              solver=self.hyperparams['solver']['choice'],
              learning_rate=self.hyperparams['solver'].get('learning_rate', 'constant'),
              learning_rate_init=self.hyperparams['solver'].get('learning_rate_init', 0.001),
              power_t=self.hyperparams['solver'].get('power_t', 0.5),
              shuffle=self.hyperparams['solver'].get('shuffle', True),
              momentum=self.hyperparams['solver'].get('momentum', 0.9),
              nesterovs_momentum=self.hyperparams['solver'].get('nesterovs_momentum', True),
              early_stopping=self.hyperparams['solver'].get('early_stopping', False),
              beta_1=self.hyperparams['solver'].get('beta_1', 0.9),
              beta_2=self.hyperparams['solver'].get('beta_2', 0.999),
              epsilon=self.hyperparams['solver'].get('epsilon', 1e-08),
              n_iter_no_change=self.hyperparams['solver'].get('n_iter_no_change', 10),
              alpha=self.hyperparams['alpha'],
              batch_size=self.hyperparams['batch_size'],
              max_iter=self.hyperparams['max_iter'],
              tol=self.hyperparams['tol'],
              warm_start=self.hyperparams['warm_start'],
              validation_fraction=self.hyperparams['validation_fraction'],
              random_state=self.random_seed,
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
                loss_=None,
                coefs_=None,
                intercepts_=None,
                n_iter_=None,
                n_layers_=None,
                n_outputs_=None,
                out_activation_=None,
                _best_coefs=None,
                _best_intercepts=None,
                _no_improvement_count=None,
                _random_state=None,
                best_validation_score_=None,
                loss_curve_=None,
                t_=None,
                _optimizer=None,
                validation_scores_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            loss_=getattr(self._clf, 'loss_', None),
            coefs_=getattr(self._clf, 'coefs_', None),
            intercepts_=getattr(self._clf, 'intercepts_', None),
            n_iter_=getattr(self._clf, 'n_iter_', None),
            n_layers_=getattr(self._clf, 'n_layers_', None),
            n_outputs_=getattr(self._clf, 'n_outputs_', None),
            out_activation_=getattr(self._clf, 'out_activation_', None),
            _best_coefs=getattr(self._clf, '_best_coefs', None),
            _best_intercepts=getattr(self._clf, '_best_intercepts', None),
            _no_improvement_count=getattr(self._clf, '_no_improvement_count', None),
            _random_state=getattr(self._clf, '_random_state', None),
            best_validation_score_=getattr(self._clf, 'best_validation_score_', None),
            loss_curve_=getattr(self._clf, 'loss_curve_', None),
            t_=getattr(self._clf, 't_', None),
            _optimizer=getattr(self._clf, '_optimizer', None),
            validation_scores_=getattr(self._clf, 'validation_scores_', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.loss_ = params['loss_']
        self._clf.coefs_ = params['coefs_']
        self._clf.intercepts_ = params['intercepts_']
        self._clf.n_iter_ = params['n_iter_']
        self._clf.n_layers_ = params['n_layers_']
        self._clf.n_outputs_ = params['n_outputs_']
        self._clf.out_activation_ = params['out_activation_']
        self._clf._best_coefs = params['_best_coefs']
        self._clf._best_intercepts = params['_best_intercepts']
        self._clf._no_improvement_count = params['_no_improvement_count']
        self._clf._random_state = params['_random_state']
        self._clf.best_validation_score_ = params['best_validation_score_']
        self._clf.loss_curve_ = params['loss_curve_']
        self._clf.t_ = params['t_']
        self._clf._optimizer = params['_optimizer']
        self._clf.validation_scores_ = params['validation_scores_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['loss_'] is not None:
            self._fitted = True
        if params['coefs_'] is not None:
            self._fitted = True
        if params['intercepts_'] is not None:
            self._fitted = True
        if params['n_iter_'] is not None:
            self._fitted = True
        if params['n_layers_'] is not None:
            self._fitted = True
        if params['n_outputs_'] is not None:
            self._fitted = True
        if params['out_activation_'] is not None:
            self._fitted = True
        if params['_best_coefs'] is not None:
            self._fitted = True
        if params['_best_intercepts'] is not None:
            self._fitted = True
        if params['_no_improvement_count'] is not None:
            self._fitted = True
        if params['_random_state'] is not None:
            self._fitted = True
        if params['best_validation_score_'] is not None:
            self._fitted = True
        if params['loss_curve_'] is not None:
            self._fitted = True
        if params['t_'] is not None:
            self._fitted = True
        if params['_optimizer'] is not None:
            self._fitted = True
        if params['validation_scores_'] is not None:
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


SKMLPRegressor.__doc__ = MLPRegressor.__doc__