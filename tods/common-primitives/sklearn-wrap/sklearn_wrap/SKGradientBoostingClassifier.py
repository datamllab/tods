from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
import sys


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
    oob_improvement_: Optional[ndarray]
    train_score_: Optional[ndarray]
    loss_: Optional[object]
    init_: Optional[object]
    estimators_: Optional[ndarray]
    n_features_: Optional[int]
    classes_: Optional[ndarray]
    max_features_: Optional[int]
    n_classes_: Optional[Union[int, List[int]]]
    alpha: Optional[float]
    _rng: Optional[object]
    n_estimators_: Optional[int]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    loss = hyperparams.Enumeration[str](
        default='deviance',
        values=['deviance', 'exponential'],
        description='loss function to be optimized. \'deviance\' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss \'exponential\' gradient boosting recovers the AdaBoost algorithm.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    learning_rate = hyperparams.Bounded[float](
        default=0.1,
        lower=0,
        upper=None,
        description='learning rate shrinks the contribution of each tree by `learning_rate`. There is a trade-off between learning_rate and n_estimators.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_estimators = hyperparams.Bounded[int](
        default=100,
        lower=1,
        upper=None,
        description='The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_depth = hyperparams.Bounded[int](
        default=3,
        lower=0,
        upper=None,
        description='maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    criterion = hyperparams.Enumeration[str](
        default='friedman_mse',
        values=['friedman_mse', 'mse', 'mae'],
        description='The function to measure the quality of a split. Supported criteria are "friedman_mse" for the mean squared error with improvement score by Friedman, "mse" for mean squared error, and "mae" for the mean absolute error. The default value of "friedman_mse" is generally the best as it can provide a better approximation in some cases.  .. versionadded:: 0.18',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    min_samples_split = hyperparams.Union(
        configuration=OrderedDict({
            'absolute': hyperparams.Bounded[int](
                default=2,
                lower=1,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.25,
                lower=0,
                upper=1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='absolute',
        description='The minimum number of samples required to split an internal node:  - If int, then consider `min_samples_split` as the minimum number. - If float, then `min_samples_split` is a percentage and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.  .. versionchanged:: 0.18 Added float values for percentages.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    min_samples_leaf = hyperparams.Union(
        configuration=OrderedDict({
            'absolute': hyperparams.Bounded[int](
                default=1,
                lower=1,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.25,
                lower=0,
                upper=0.5,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='absolute',
        description='The minimum number of samples required to be at a leaf node:  - If int, then consider `min_samples_leaf` as the minimum number. - If float, then `min_samples_leaf` is a percentage and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.  .. versionchanged:: 0.18 Added float values for percentages.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    min_weight_fraction_leaf = hyperparams.Bounded[float](
        default=0,
        lower=0,
        upper=0.5,
        description='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    subsample = hyperparams.Bounded[float](
        default=1.0,
        lower=0,
        upper=None,
        description='The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. `subsample` interacts with the parameter `n_estimators`. Choosing `subsample < 1.0` leads to a reduction of variance and an increase in bias.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_features = hyperparams.Union(
        configuration=OrderedDict({
            'specified_int': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                default=0,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'calculated': hyperparams.Enumeration[str](
                values=['auto', 'sqrt', 'log2'],
                default='auto',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.25,
                lower=0,
                upper=1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split. - If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at each split. - If "auto", then `max_features=sqrt(n_features)`. - If "sqrt", then `max_features=sqrt(n_features)`. - If "log2", then `max_features=log2(n_features)`. - If None, then `max_features=n_features`.  Choosing `max_features < n_features` leads to a reduction of variance and an increase in bias.  Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_leaf_nodes = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Bounded[int](
                default=10,
                lower=0,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    min_impurity_decrease = hyperparams.Bounded[float](
        default=0.0,
        lower=0.0,
        upper=None,
        description='A node will be split if this split induces a decrease of the impurity greater than or equal to this value.  The weighted impurity decrease equation is the following::  N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)  where ``N`` is the total number of samples, ``N_t`` is the number of samples at the current node, ``N_t_L`` is the number of samples in the left child, and ``N_t_R`` is the number of samples in the right child.  ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum, if ``sample_weight`` is passed.  .. versionadded:: 0.19',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    warm_start = hyperparams.UniformBool(
        default=False,
        description='When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    presort = hyperparams.Union(
        configuration=OrderedDict({
            'bool': hyperparams.UniformBool(
                default=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'auto': hyperparams.Constant(
                default='auto',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='auto',
        description='Whether to presort the data to speed up the finding of best splits in fitting. Auto mode by default will use presorting on dense data and default to normal sorting on sparse data. Setting presort to true on sparse data will raise an error.  .. versionadded:: 0.17 *presort* parameter.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    validation_fraction = hyperparams.Bounded[float](
        default=0.1,
        lower=0,
        upper=1,
        description='The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if ``n_iter_no_change`` is set to an integer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_iter_no_change = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Bounded[int](
                default=5,
                lower=0,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='``n_iter_no_change`` is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside ``validation_fraction`` size of the training data as validation and terminate training when validation score is not improving in all of the previous ``n_iter_no_change`` numbers of iterations.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    tol = hyperparams.Bounded[float](
        default=0.0001,
        lower=0,
        upper=None,
        description='Tolerance for the early stopping. When the loss is not improving by at least tol for ``n_iter_no_change`` iterations (if set to a number), the training stops.',
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

class SKGradientBoostingClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                          ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn GradientBoostingClassifier
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.GRADIENT_BOOSTING, ],
         "name": "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier",
         "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
         "python_path": "d3m.primitives.classification.gradient_boosting.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html']},
         "version": "2019.11.13",
         "id": "01d2c086-91bf-3ca5-b023-5139cf239c77",
         "hyperparams_to_tune": ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'max_features'],
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
        self._clf = GradientBoostingClassifier(
              loss=self.hyperparams['loss'],
              learning_rate=self.hyperparams['learning_rate'],
              n_estimators=self.hyperparams['n_estimators'],
              max_depth=self.hyperparams['max_depth'],
              criterion=self.hyperparams['criterion'],
              min_samples_split=self.hyperparams['min_samples_split'],
              min_samples_leaf=self.hyperparams['min_samples_leaf'],
              min_weight_fraction_leaf=self.hyperparams['min_weight_fraction_leaf'],
              subsample=self.hyperparams['subsample'],
              max_features=self.hyperparams['max_features'],
              max_leaf_nodes=self.hyperparams['max_leaf_nodes'],
              min_impurity_decrease=self.hyperparams['min_impurity_decrease'],
              warm_start=self.hyperparams['warm_start'],
              presort=self.hyperparams['presort'],
              validation_fraction=self.hyperparams['validation_fraction'],
              n_iter_no_change=self.hyperparams['n_iter_no_change'],
              tol=self.hyperparams['tol'],
              verbose=_verbose,
              random_state=self.random_seed,
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
                oob_improvement_=None,
                train_score_=None,
                loss_=None,
                init_=None,
                estimators_=None,
                n_features_=None,
                classes_=None,
                max_features_=None,
                n_classes_=None,
                alpha=None,
                _rng=None,
                n_estimators_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            oob_improvement_=getattr(self._clf, 'oob_improvement_', None),
            train_score_=getattr(self._clf, 'train_score_', None),
            loss_=getattr(self._clf, 'loss_', None),
            init_=getattr(self._clf, 'init_', None),
            estimators_=getattr(self._clf, 'estimators_', None),
            n_features_=getattr(self._clf, 'n_features_', None),
            classes_=getattr(self._clf, 'classes_', None),
            max_features_=getattr(self._clf, 'max_features_', None),
            n_classes_=getattr(self._clf, 'n_classes_', None),
            alpha=getattr(self._clf, 'alpha', None),
            _rng=getattr(self._clf, '_rng', None),
            n_estimators_=getattr(self._clf, 'n_estimators_', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.oob_improvement_ = params['oob_improvement_']
        self._clf.train_score_ = params['train_score_']
        self._clf.loss_ = params['loss_']
        self._clf.init_ = params['init_']
        self._clf.estimators_ = params['estimators_']
        self._clf.n_features_ = params['n_features_']
        self._clf.classes_ = params['classes_']
        self._clf.max_features_ = params['max_features_']
        self._clf.n_classes_ = params['n_classes_']
        self._clf.alpha = params['alpha']
        self._clf._rng = params['_rng']
        self._clf.n_estimators_ = params['n_estimators_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['oob_improvement_'] is not None:
            self._fitted = True
        if params['train_score_'] is not None:
            self._fitted = True
        if params['loss_'] is not None:
            self._fitted = True
        if params['init_'] is not None:
            self._fitted = True
        if params['estimators_'] is not None:
            self._fitted = True
        if params['n_features_'] is not None:
            self._fitted = True
        if params['classes_'] is not None:
            self._fitted = True
        if params['max_features_'] is not None:
            self._fitted = True
        if params['n_classes_'] is not None:
            self._fitted = True
        if params['alpha'] is not None:
            self._fitted = True
        if params['_rng'] is not None:
            self._fitted = True
        if params['n_estimators_'] is not None:
            self._fitted = True


    def log_likelihoods(self, *,
                    outputs: Outputs,
                    inputs: Inputs,
                    timeout: float = None,
                    iterations: int = None) -> CallResult[Sequence[float]]:
        inputs = inputs.iloc[:, self._training_indices]  # Get ndarray
        outputs = outputs.iloc[:, self._target_column_indices]

        if len(inputs.columns) and len(outputs.columns):

            if outputs.shape[1] != self._clf.n_outputs_:
                raise exceptions.InvalidArgumentValueError("\"outputs\" argument does not have the correct number of target columns.")

            log_proba = self._clf.predict_log_proba(inputs)

            # Making it always a list, even when only one target.
            if self._clf.n_outputs_ == 1:
                log_proba = [log_proba]
                classes = [self._clf.classes_]
            else:
                classes = self._clf.classes_

            samples_length = inputs.shape[0]

            log_likelihoods = []
            for k in range(self._clf.n_outputs_):
                # We have to map each class to its internal (numerical) index used in the learner.
                # This allows "outputs" to contain string classes.
                outputs_column = outputs.iloc[:, k]
                classes_map = pandas.Series(numpy.arange(len(classes[k])), index=classes[k])
                mapped_outputs_column = outputs_column.map(classes_map)

                # For each target column (column in "outputs"), for each sample (row) we pick the log
                # likelihood for a given class.
                log_likelihoods.append(log_proba[k][numpy.arange(samples_length), mapped_outputs_column])

            results = d3m_dataframe(dict(enumerate(log_likelihoods)), generate_metadata=True)
            results.columns = outputs.columns

            for k in range(self._clf.n_outputs_):
                column_metadata = outputs.metadata.query_column(k)
                if 'name' in column_metadata:
                    results.metadata = results.metadata.update_column(k, {'name': column_metadata['name']})

        else:
            results = d3m_dataframe(generate_metadata=True)

        return CallResult(results)
    


    def produce_feature_importances(self, *, timeout: float = None, iterations: int = None) -> CallResult[d3m_dataframe]:
        output = d3m_dataframe(self._clf.feature_importances_.reshape((1, len(self._input_column_names))))
        output.columns = self._input_column_names
        for i in range(len(self._input_column_names)):
            output.metadata = output.metadata.update_column(i, {"name": self._input_column_names[i]})
        return CallResult(output)
    
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


SKGradientBoostingClassifier.__doc__ = GradientBoostingClassifier.__doc__