import os
from collections import OrderedDict
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.multioutput import MultiOutputClassifier  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult, ProbabilisticCompositionalityMixin, SamplingCompositionalityMixin, ContinueFitMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import common_primitives

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    estimators: Optional[Union[xgb.XGBClassifier, List[xgb.XGBClassifier]]]
    booster: Optional[Union[xgb.Booster, List[xgb.Booster]]]
    classes: Optional[Union[np.ndarray, List[np.ndarray]]]
    n_classes: Optional[Union[int, List[int]]]
    objective: Optional[str]
    multi_output_estimator_dict: Optional[Dict]
    attribute_columns_names: Optional[List[str]]
    target_columns_metadata: Optional[List[OrderedDict]]
    target_columns_names: Optional[List[str]]
    le: Optional[LabelEncoder]


class Hyperparams(hyperparams.Hyperparams):
    n_estimators = hyperparams.UniformInt(
        lower=1,
        upper=10000,
        default=200,
        description='The number of trees in the forest.',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    )
    n_more_estimators = hyperparams.UniformInt(
        lower=1,
        upper=10000,
        default=100,
        description='When continuing a fit, it controls how many more trees to add every time.',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    )
    max_depth = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=3,
            ),
            unlimited=hyperparams.Enumeration[int](
                values=[0],
                default=0,
                description='Nodes are expanded until all leaves are pure or until all leaves contain less than "min_samples_split" samples.',
            ),
        ),
        default='limit',
        description='The maximum depth of the tree.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    learning_rate = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.1,
        description=r'Boosting learning rate (xgb\`s \"eta\")',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    gamma = hyperparams.Bounded[float](
        lower=0.0,
        upper=None,
        default=0.0,
        description='Minimum loss reduction required to make a further partition on a leaf node of the tree',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_child_weight = hyperparams.Bounded[int](
        lower=0,
        upper=None,
        default=1,
        description='Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results '
                    'in a leaf node with the sum of instance weight less than min_child_weight, then the building '
                    'process will give up further partitioning ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    max_delta_step = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=1,
                # TODO: 1-10 instead?
                upper=None,
                default=1,
                description='Maximum delta step we allow each leaf output to be.'
            ),
            unlimited=hyperparams.Enumeration[int](
                values=[0],
                default=0,
                description='No constraint.',
            ),
        ),
        default='unlimited',
        description='Maximum delta step we allow.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    # TODO: better way to represent lower bound is exclusive?
    subsample = hyperparams.Uniform(
        lower=0.0001,
        upper=1,
        default=1,
        upper_inclusive=True,
        description='Subsample ratio of the training instances,this will prevent overfitting. Subsampling will occur '
                    'once in every boosting iteration.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    colsample_bytree = hyperparams.Uniform(
        lower=0.0001,
        upper=1,
        default=1,
        upper_inclusive=True,
        description='Subsample ratio of columns when constructing each tree. Subsampling will occur once in every '
                    'boosting iteration',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    colsample_bylevel = hyperparams.Uniform(
        lower=0.0001,
        upper=1,
        default=1,
        upper_inclusive=True,
        description='Subsample ratio of columns for each split, in each level. Subsampling will occur each time a new '
                    'split is made',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    reg_lambda = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=1,
        description='L2 regularization term on weights. Increasing this value will make model more conservative.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    reg_alpha = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0,
        description='L1 regularization term on weights. Increasing this value will make model more conservative.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    scale_pos_weight = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=1,
        description='Control the balance of positive and negative weights, useful for unbalanced classes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    base_score = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0.5,
        description='The initial prediction score of all instances, global bias.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    n_jobs = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=1,
            ),
            all_cores=hyperparams.Enumeration[int](
                values=[-1],
                default=-1,
                description='The number of jobs is set to the number of cores.',
            ),
        ),
        default='limit',
        description='The number of jobs to run in parallel for both "fit" and "produce".',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter'],
    )
    sample_type = hyperparams.Enumeration[str](
        values=['uniform', 'weighted'],
        default='uniform',
        description='Type of sampling algorithm',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    normalize_type = hyperparams.Enumeration[str](
        values=['tree', 'forest'],
        default='tree',
        description='Type of normalization algorithm',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    rate_drop = hyperparams.Bounded[float](
        lower=0,
        upper=1.0,
        default=0.0,
        description='Dropout rate (a fraction of previous trees to drop during the dropout)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    one_drop = hyperparams.Enumeration[int](
        values=[0, 1],
        default=0,
        description='When this flag is enabled, at least one tree is always dropped during the dropout',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    skip_drop = hyperparams.Bounded[float](
        lower=0,
        upper=1.0,
        default=0.0,
        description='Probability of skipping the dropout procedure during a boosting iteration',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    use_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of inputs column indices to force primitive to operate on. If any specified column cannot be used, it is skipped.",
    )
    exclude_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of inputs column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    use_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of outputs column indices to force primitive to operate on. If any specified column cannot be used, it is skipped.",
    )
    exclude_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of outputs column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        # Default value depends on the nature of the primitive.
        default='append',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should resulting columns be appended, should they replace original columns, or should only resulting columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


# TODO: Instead of using XGBoostClassifier instance provided by the xgboost's sklearn interface, use XGBoost original
#  API to prevent ugly set and get params
class XGBoostDartClassifierPrimitive(ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                     SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                     ContinueFitMixin[Inputs, Outputs, Params, Hyperparams],
                                     SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A XGBoost classifier using ``xgb.XGBoostClassifier`` with Dart Boosting type.

    It uses semantic types to determine which columns to operate on.
    """

    __author__ = 'TAMU DARPA D3M Team, TsungLin Yang <lin.yang@tamu.edu>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7476950e-4373-4cf5-a852-7e16afb8e098',
            'version': '0.1.0',
            'name': "XGBoost DART classifier",
            'python_path': 'd3m.primitives.classification.xgboost_dart.Common',
            'keywords': ['xgboost', 'decision tree', 'gradient boosted trees', ],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:lin.yang@tamu.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.GRADIENT_BOOSTING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
            'hyperparams_to_tune': [
                'learning_rate',
                'colsample_bytree',
                'min_child_weight',
                'subsample',
                'max_depth',
                'max_delta_step'
            ]
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, _verbose: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # We need random seed multiple times (every time an underlying "RandomForestClassifier" is instantiated),
        # and when we sample. So instead we create our own random state we use everywhere.
        self._random_state = np.random.RandomState(self.random_seed)
        self._verbose = _verbose
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        self._new_training_data = False
        self._learner: Union[xgb.XGBClassifier, MultiOutputClassifier] = None
        self._attribute_columns_names: List[str] = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._target_columns_names: List[str] = None
        self._multi_output_estimator_dict: Dict = {}

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._new_training_data = True

    def _create_learner(self) -> None:
        self._learner = xgb.XGBClassifier(
            max_depth=self.hyperparams['max_depth'],
            learning_rate=self.hyperparams['learning_rate'],
            n_estimators=self.hyperparams['n_estimators'],
            gamma=self.hyperparams['gamma'],
            min_child_weight=self.hyperparams['min_child_weight'],
            max_delta_step=self.hyperparams['max_delta_step'],
            subsample=self.hyperparams['subsample'],
            colsample_bylevel=self.hyperparams['colsample_bylevel'],
            colsample_bytree=self.hyperparams['colsample_bytree'],
            reg_alpha=self.hyperparams['reg_alpha'],
            reg_lambda=self.hyperparams['reg_lambda'],
            scale_pos_weight=self.hyperparams['scale_pos_weight'],
            base_score=self.hyperparams['base_score'],
            n_jobs=-1 if self.hyperparams['n_jobs'] is None else self.hyperparams['n_jobs'],
            random_state=self.random_seed,
            booster='dart',
            silent=not bool(self._verbose)
        )

    def _get_target_columns_metadata(self, outputs_metadata: metadata_base.DataMetadata) -> List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = list(column_metadata.get('semantic_types', []))
            if 'https://metadata.datadrivendiscovery.org/types/PredictedTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            semantic_types = [semantic_type for semantic_type in semantic_types if
                              semantic_type != 'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            column_metadata['semantic_types'] = semantic_types

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    def _store_target_columns_metadata(self, inputs: Inputs, outputs: Outputs) -> None:
        self._attribute_columns_names = list(inputs.columns)
        self._target_columns_metadata = self._get_target_columns_metadata(outputs.metadata)
        self._target_columns_names = list(outputs.columns)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None or self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        # An optimization. Do not refit if data has not changed.
        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        inputs, _ = self._select_inputs_columns(self._training_inputs)
        outputs, _ = self._select_outputs_columns(self._training_outputs)

        self._create_learner()

        # A special case for sklearn. It prefers an 1D array instead of 2D when there is only one target.
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            fit_outputs = np.ravel(outputs)
        else:
            fit_outputs = outputs
            self._learner = MultiOutputClassifier(self._learner)

        self._learner.fit(np.array(inputs), np.array(fit_outputs))
        if isinstance(self._learner, MultiOutputClassifier):
            for _, output in outputs.iteritems():
                estimator_index = next((index for index, estimator in enumerate(self._learner.estimators_) if
                                        set(output.unique()) == set(estimator.classes_)), None)
                self._multi_output_estimator_dict[output.name] = estimator_index

        self._store_target_columns_metadata(inputs, outputs)

        return CallResult(None)

    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None or self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        # This model is not improving fitting if called multiple times on the same data.
        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        if not self._learner:
            self._create_learner()

        inputs, _ = self._select_inputs_columns(self._training_inputs)
        outputs, _ = self._select_outputs_columns(self._training_outputs)

        #  using xgboost api to continue fit the classifier.
        def continue_xgb_booster(xgb_model: xgb.XGBClassifier, inputs: Inputs,
                                 output_values: Union[np.ndarray, Outputs], num_of_boosting_round: int) -> None:
            label = LabelEncoder().fit_transform(output_values)
            dtrain = xgb.DMatrix(data=inputs.values, label=label)
            model_param = xgb_model.get_xgb_params()
            del (model_param['n_estimators'])
            model_param['objective'] = xgb_model.objective
            model_param['num_class'] = xgb_model.n_classes_
            booster = xgb.train(params=model_param, dtrain=dtrain,
                                num_boost_round=num_of_boosting_round,
                                xgb_model=xgb_model.get_booster())
            xgb_model.set_params(_Booster=booster)

        # A special case for sklearn. It prefers an 1D array instead of 2D when there is only one target.
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            continue_xgb_booster(self._learner, inputs, np.ravel(outputs), self.hyperparams['n_more_estimators'])
        else:
            # TODO Currently doesn't support unseen target for continuing multi-output classification.
            if outputs.shape[1] != len(self._learner.estimators_):
                raise exceptions.InvalidArgumentValueError("Number of target does not match with the original data")
            for _, output in outputs.iteritems():
                estimator_index = self._multi_output_estimator_dict.get(output.name, None)
                if estimator_index is None:
                    raise exceptions.InvalidArgumentValueError(
                        'Unseen target column when continuing fit {}'.format(output.name))
                estimator = self._learner.estimators_[self._multi_output_estimator_dict[output.name]]
                continue_xgb_booster(estimator, inputs, output, self.hyperparams['n_more_estimators'])

        self._store_target_columns_metadata(inputs, outputs)

        return CallResult(None)

    def _update_predictions_metadata(self, outputs: Optional[Outputs], target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
        outputs_metadata = metadata_base.DataMetadata()
        if outputs is not None:
            outputs_metadata = outputs_metadata.generate(outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, predictions: np.ndarray) -> Outputs:
        outputs = container.DataFrame(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(outputs, self._target_columns_metadata)
        outputs.columns = self._target_columns_names
        return outputs

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        selected_inputs, columns_to_use = self._select_inputs_columns(inputs)

        # TODO: If the booster object is DART type, predict() will perform dropouts, i.e. only some of the trees will
        #  be evaluated. This will produce incorrect results if data is not the training data. To obtain correct
        #  results on test sets, set ntree_limit to a nonzero value.
        # Potential BUG
        if not isinstance(self._learner, MultiOutputClassifier):
            predictions = self._learner.predict(selected_inputs.values, ntree_limit=self._learner.n_estimators)
        else:
            predictions = []
            for estimator in self._learner.estimators_:
                predictions.append(estimator.predict(selected_inputs.values, ntree_limit=estimator.n_estimators))
            predictions = np.array(predictions).transpose()
        output_columns = [self._wrap_predictions(predictions)]

        outputs = base_utils.combine_columns(
            inputs, columns_to_use, output_columns,
            return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'],
        )

        return CallResult(outputs)

    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> \
            CallResult[Sequence[Outputs]]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        inputs, _ = self._select_inputs_columns(inputs)

        samples = []
        for i in range(num_samples):
            predictions = self._learner.predict(inputs.values)
            samples.append(self._wrap_predictions(predictions))

        return CallResult(samples)

    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> \
            CallResult[Outputs]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        outputs, _ = self._select_outputs_columns(outputs)
        inputs, _ = self._select_inputs_columns(inputs)
        log_proba = np.log(self._learner.predict_proba(np.array(inputs)))

        if outputs.shape[1] == 1:
            log_proba = [log_proba]
            classes = [self._learner.classes_]
        else:
            classes = [x.classes_ for x in self._learner.estimators_]

        samples_length = inputs.shape[0]

        log_likelihoods = []
        for k in range(outputs.shape[1]):
            # We have to map each class to its internal (numerical) index used in the learner.
            # This allows "outputs" to contain string classes.
            outputs_column = outputs.iloc[:, k]
            classes_map = pd.Series(np.arange(len(classes[k])), index=classes[k])
            mapped_outputs_column = outputs_column.map(classes_map)

            # For each target column (column in "outputs"), for each sample (row) we pick the log
            # likelihood for a given class.
            log_likelihoods.append(log_proba[k][np.arange(samples_length), mapped_outputs_column])

        results = container.DataFrame(dict(enumerate(log_likelihoods)), generate_metadata=True)
        results.columns = outputs.columns

        for k in range(outputs.shape[1]):
            column_metadata = outputs.metadata.query_column(k)
            if 'name' in column_metadata:
                results.metadata = results.metadata.update_column(k, {'name': column_metadata['name']})

        return CallResult(results)

    def get_params(self) -> Params:
        if not self._learner:
            return Params(
                estimators=None,
                booster=None,
                classes=None,
                n_classes=None,
                objective=None,
                multi_output_estimator_dict=None,
                le=None,
                target_columns_metadata=None,
                attribute_columns_names=None,
                target_columns_names=None
            )

        return Params(
            estimators=self._learner.estimators_ if isinstance(self._learner, MultiOutputClassifier) else self._learner,
            booster=self._learner.get_booster() if not isinstance(self._learner, MultiOutputClassifier) else [
                estimator.get_booster() for estimator in self._learner.estimators_],
            classes=self._learner.classes_
            if not isinstance(self._learner, MultiOutputClassifier) else [estimator.classes_ for estimator in
                                                                          self._learner.estimators_],
            n_classes=self._learner.n_classes_
            if not isinstance(self._learner, MultiOutputClassifier) else [estimator.n_classes_ for estimator in
                                                                          self._learner.estimators_],
            objective=self._learner.objective
            if not isinstance(self._learner, MultiOutputClassifier) else self._learner.estimators_[0].objective,
            multi_output_estimator_dict=self._multi_output_estimator_dict
            if isinstance(self._learner, MultiOutputClassifier) else {},
            attribute_columns_names=self._attribute_columns_names,
            target_columns_metadata=self._target_columns_metadata,
            target_columns_names=self._target_columns_names,
            le=self._learner._le if not isinstance(self._learner, MultiOutputClassifier) else None
        )

    def set_params(self, *, params: Params) -> None:
        if not all(params[param] is not None for param in
                   ['estimators', 'booster', 'classes', 'n_classes', 'objective', 'target_columns_metadata']):
            self._learner = None
        else:
            if isinstance(self._learner, MultiOutputClassifier):
                self._learner.estimators_ = params['estimators']
                self._multi_output_estimator_dict = params['multi_output_estimator_dict']
            else:
                self._create_learner()
                # A little hack to set lable encoder of XGBoostClassifier instance to prevent exceptions.
                self._learner._le = params['le']
                # Another hack to make sure class attribute out side __init__ gets set properly
                self._learner.classes_ = 0
                self._learner.n_classes_ = 0
                self._learner.set_params(_Booster=params['booster'], n_classes_=params['n_classes'],
                                         classes_=params['classes'], objective=params['objective'])
            self._attribute_columns_names = params['attribute_columns_names']
            self._target_columns_metadata = params['target_columns_metadata']
            self._target_columns_names = params['target_columns_names']

    def __getstate__(self) -> dict:
        state = super().__getstate__()

        # Random state is not part of the "Params", but it is part of the state we want to
        # pickle and unpickle to have full reproducibility. So we have to add it ourselves here.
        # This is also difference between pickling/unpickling and "get_params"/"set_params".
        # The later saves only the model state which is useful to produce at a later time, but
        # if we want to also reproduce the exact sequence of values, we should be using pickling.
        state['random_state'] = self._random_state

        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self._random_state = state['random_state']

    def _can_use_inputs_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        if not d3m_utils.is_numeric(column_metadata['structural_type']):
            return False

        return 'https://metadata.datadrivendiscovery.org/types/Attribute' in column_metadata.get('semantic_types', [])

    def _get_inputs_columns(self, inputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_inputs_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            inputs_metadata,
            self.hyperparams['use_inputs_columns'],
            self.hyperparams['exclude_inputs_columns'],
            can_use_column,
        )

        if not columns_to_use:
            raise ValueError("No inputs columns.")

        if self.hyperparams['use_inputs_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified inputs columns can used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _can_use_outputs_column(self, outputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        return 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in column_metadata.get('semantic_types', [])

    def _get_outputs_columns(self, outputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_outputs_column(outputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            outputs_metadata,
            self.hyperparams['use_outputs_columns'],
            self.hyperparams['exclude_outputs_columns'],
            can_use_column,
        )

        if not columns_to_use:
            raise ValueError("No outputs columns.")

        if self.hyperparams['use_outputs_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified outputs columns can used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _select_inputs_columns(self, inputs: Inputs) -> Tuple[Inputs, List[int]]:
        columns_to_use = self._get_inputs_columns(inputs.metadata)

        return inputs.select_columns(columns_to_use), columns_to_use

    def _select_outputs_columns(self, outputs: Outputs) -> Tuple[Outputs, List[int]]:
        columns_to_use = self._get_outputs_columns(outputs.metadata)

        return outputs.select_columns(columns_to_use), columns_to_use
