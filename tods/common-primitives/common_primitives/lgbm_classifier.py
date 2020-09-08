import math
import os
from collections import OrderedDict
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple, Callable

import lightgbm as lgbm  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.multioutput import MultiOutputClassifier  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult, ProbabilisticCompositionalityMixin, SamplingCompositionalityMixin, \
    ContinueFitMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import common_primitives

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    booster: Optional[Union[lgbm.basic.Booster, List[lgbm.basic.Booster]]]
    estimators: Optional[Union[List[lgbm.LGBMClassifier], lgbm.LGBMClassifier]]
    classes: Optional[Union[np.ndarray, List[np.ndarray]]]
    n_classes: Optional[Union[int, List[int]]]
    n_features: Optional[Union[int, List[int]]]
    objective: Optional[Union[str, Callable]]
    multi_output_estimator_dict: Optional[Dict]
    target_columns_names: Optional[List[str]]
    target_columns_metadata: Optional[List[OrderedDict]]
    le: Optional[LabelEncoder]
    attribute_columns_names: Optional[List[str]]


class Hyperparams(hyperparams.Hyperparams):
    n_estimators = hyperparams.UniformInt(
        lower=1,
        upper=10000,
        default=100,
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
                default=5,
            ),
            unlimited=hyperparams.Enumeration[int](
                values=[-1],
                default=-1,
            ),
        ),
        default='limit',
        description='The maximum depth of the tree.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    num_leaves_base = hyperparams.Bounded[float](
        lower=1,
        upper=2,
        default=2,
        description='Maximum tree leaves for base learners, this value is the base of the formula num_leaves_base^(max_depth)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    subsample_for_bin = hyperparams.Bounded[int](
        lower=1,
        upper=None,
        default=200000,
        description='number of data that sampled to construct histogram bins',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    learning_rate = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.1,
        description=r'Boosting learning rate (xgb\`s \"eta\")',
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
    min_child_samples = hyperparams.Bounded[int](
        lower=0,
        upper=None,
        default=20,
        description='minimal number of data in one leaf. Can be used to deal with over-fitting',
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
    subsample_freq = hyperparams.Bounded[int](
        lower=0,
        upper=1,
        default=0,
        description='frequency for bagging',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    colsample_bytree = hyperparams.Bounded[float](
        lower=0,
        upper=1,
        default=1,
        description='Subsample ratio of columns when constructing each tree. Subsampling will occur once in every '
                    'boosting iteration',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_split_gain = hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0,
        description='the minimal gain to perform split',
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


class LightGBMClassifierPrimitive(ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                  SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                  ContinueFitMixin[Inputs, Outputs, Params, Hyperparams],
                                  SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A lightGBM classifier using ``lgbm.LGBMClassifier``.

    It uses semantic types to determine which columns to operate on.
    """
    __author__ = 'TAMU DARPA D3M Team, TsungLin Yang <lin.yang@tamu.edu>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '259aa747-795c-435e-8e33-8c32a4c83c6b',
            'version': '0.1.0',
            'name': "LightGBM GBTree classifier",
            'python_path': 'd3m.primitives.classification.light_gbm.Common',
            'keywords': ['lightgbm', 'decision tree', 'gradient boosted trees', ],
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
        self._learner: Union[lgbm.LGBMClassifier, MultiOutputClassifier] = None
        self._multi_output_estimator_dict: Dict = {}
        self._target_columns_metadata: List[OrderedDict] = None
        self._attribute_columns_names: List[str] = None
        self._target_columns_names: List[str] = None

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._new_training_data = True

    def _create_learner(self) -> None:
        # TODO: temporarily deal with the dependency between max_depth and num_leaves. When max_depth is not limited,
        #  set num_leaves to default value 31
        num_leaves = math.floor(pow(self.hyperparams['num_leaves_base'], self.hyperparams['max_depth'])) if \
            self.hyperparams['max_depth'] else 31
        self._learner = lgbm.LGBMClassifier(
            n_estimators=self.hyperparams['n_estimators'],
            max_depth=self.hyperparams['max_depth'],
            num_leaves=num_leaves,
            subsample_for_bin=self.hyperparams['subsample_for_bin'],
            learning_rate=self.hyperparams['learning_rate'],
            min_child_weight=self.hyperparams['min_child_weight'],
            min_child_samples=self.hyperparams['min_child_samples'],
            max_delta_step=self.hyperparams['max_delta_step'],
            subsample=self.hyperparams['subsample'],
            subsample_freq=self.hyperparams['subsample_freq'],
            min_split_gain=self.hyperparams['min_split_gain'],
            colsample_bytree=self.hyperparams['colsample_bytree'],
            reg_lambda=self.hyperparams['reg_lambda'],
            reg_alpha=self.hyperparams['reg_alpha'],
            n_jobs=-1 if self.hyperparams['n_jobs'] is None else self.hyperparams['n_jobs'],
            random_state=self.random_seed,
            boosting_type='gbdt',
            verbose=self._verbose - 1
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

    def _cast_to_category_type(self, data: container.DataFrame) -> container.DataFrame:
        cat_cols = data.metadata.get_columns_with_semantic_type(
            'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        if cat_cols:
            data.iloc[:, cat_cols] = data.iloc[:, cat_cols].astype('category')
        return data

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None or self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        # An optimization. Do not refit if data has not changed.
        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        inputs, _ = self._select_inputs_columns(self._training_inputs)
        outputs, _ = self._select_outputs_columns(self._training_outputs)
        # cast categorical feature column to pandas category type
        inputs = self._cast_to_category_type(inputs)
        self._create_learner()
        # A special case for sklearn. It prefers an 1D array instead of 2D when there is only one target.
        if outputs.shape[1] > 1:
            raise exceptions.InvalidArgumentValueError('Multioutput is not supported by LGBM classifier primitive')

        self._learner.fit(inputs, outputs)

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
        inputs = self._cast_to_category_type(inputs)
        outputs, _ = self._select_outputs_columns(self._training_outputs)

        #  using lightgbm api to continue fit the classifier.
        def continue_lgb_booster(lgb_model: lgbm.LGBMClassifier, inputs: Inputs,
                                 output_values: Union[np.ndarray, Outputs], num_of_boosting_round: int) -> None:
            label = LabelEncoder().fit_transform(output_values)
            train_set = lgbm.Dataset(data=inputs, label=label)
            model_param = lgb_model.get_params()

            del model_param['n_estimators'], model_param['silent'], model_param['importance_type']
            model_param['objective'] = lgb_model.objective_
            model_param['num_class'] = lgb_model.n_classes_
            booster = lgbm.train(params=model_param, train_set=train_set,
                                 num_boost_round=num_of_boosting_round,
                                 init_model=lgb_model.booster_, keep_training_booster=True)
            lgb_model.set_params(Booster=booster)

        # A special case for sklearn. It prefers an 1D array instead of 2D when there is only one target.
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            continue_lgb_booster(self._learner, inputs, np.ravel(outputs), self.hyperparams['n_more_estimators'])
        else:
            raise exceptions.InvalidArgumentValueError('Multioutput is not supported by LGBM classifier primitive')
            # # TODO Currently doesn't support unseen target for continuing multi-output classification.

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
        selected_inputs = self._cast_to_category_type(selected_inputs)

        predictions = self._learner.predict(selected_inputs)

        output_columns = [self._wrap_predictions(predictions)]

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns,
                                             return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'])

        return CallResult(outputs)

    def produce_feature_importances(self, *, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")
        # TODO: is feature importances the same for every target?
        feature_importances = self._learner.feature_importances_
        feature_importances = feature_importances / sum(feature_importances)
        feature_importances_array = feature_importances.reshape((1, len(self._attribute_columns_names)))

        feature_importances = container.DataFrame(feature_importances_array, generate_metadata=True)
        feature_importances.columns = self._attribute_columns_names
        for k in range(len(self._attribute_columns_names)):
            feature_importances.metadata = feature_importances.metadata.update_column(k, {
                'name': self._attribute_columns_names[k]})

        return CallResult(feature_importances)

    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> \
            CallResult[Sequence[Outputs]]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        inputs, _ = self._select_inputs_columns(inputs)
        inputs = self._cast_to_category_type(inputs)

        samples = []
        for i in range(num_samples):
            predictions = self._learner.predict(inputs)
            samples.append(self._wrap_predictions(predictions))

        return CallResult(samples)

    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> \
            CallResult[Outputs]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        outputs, _ = self._select_outputs_columns(outputs)
        inputs, _ = self._select_inputs_columns(inputs)
        inputs = self._cast_to_category_type(inputs)
        log_proba = np.log(self._learner.predict_proba(inputs))

        if outputs.shape[1] > 1:
            raise exceptions.InvalidArgumentValueError('Multioutput is not supported by LGBM classifier primitive')

        samples_length = inputs.shape[0]

        # We have to map each class to its internal (numerical) index used in the learner.
        # This allows "outputs" to contain string classes.
        outputs_column = outputs.iloc[:, 0]
        classes_map = pd.Series(np.arange(len(self._learner.classes_)), index=self._learner.classes_)
        mapped_outputs_column = outputs_column.map(classes_map)

        # For each target column (column in "outputs"), for each sample (row) we pick the log
        # likelihood for a given class.
        log_likelihood = log_proba[np.arange(samples_length), mapped_outputs_column]

        result = container.DataFrame(log_likelihood, generate_metadata=True)
        result.columns = outputs.columns

        column_metadata = outputs.metadata.query_column(0)
        if 'name' in column_metadata:
            result.metadata = result.metadata.update_column(0, {'name': column_metadata['name']})

        return CallResult(result)

    def get_params(self) -> Params:
        if not self._learner:
            return Params(
                estimators=None,
                booster=None,
                classes=None,
                n_classes=None,
                n_features=None,
                objective=None,
                multi_output_estimator_dict=None,
                target_columns_metadata=None,
            )

        return Params(
            estimators=self._learner.estimators_ if isinstance(self._learner, MultiOutputClassifier) else self._learner,
            booster=self._learner.booster_ if not isinstance(self._learner, MultiOutputClassifier) else [
                estimator.booster_ for estimator in self._learner.estimators_],
            classes=self._learner.classes_
            if not isinstance(self._learner, MultiOutputClassifier) else [estimator.classes_ for estimator in
                                                                          self._learner.estimators_],
            n_classes=self._learner.n_classes_
            if not isinstance(self._learner, MultiOutputClassifier) else [estimator.n_classes_ for estimator in
                                                                          self._learner.estimators_],
            n_features=self._learner.n_features_
            if not isinstance(self._learner, MultiOutputClassifier) else [estimator.n_features_ for estimator in
                                                                          self._learner.estimators_],
            objective=self._learner.objective_
            if not isinstance(self._learner, MultiOutputClassifier) else self._learner.estimators_[0].objective,
            multi_output_estimator_dict=self._multi_output_estimator_dict
            if isinstance(self._learner, MultiOutputClassifier) else {},
            target_columns_names=self._target_columns_names,
            attribute_columns_names=self._attribute_columns_names,
            target_columns_metadata=self._target_columns_metadata,
            le=self._learner._le if not isinstance(self._learner, MultiOutputClassifier) else None

        )

    def set_params(self, *, params: Params) -> None:
        if not all(params[param] is not None for param in
                   ['booster', 'objective', 'classes', 'n_classes', 'n_features', 'target_columns_metadata']) or \
                not params['estimators']:
            self._learner = None
        else:
            if isinstance(self._learner, MultiOutputClassifier):
                self._learner.estimators_ = params['estimators']
                self._multi_output_estimator_dict = params['multi_output_estimator_dict']
            else:
                self._create_learner()
                lgbm_param = params.copy()
                del lgbm_param['estimators'], lgbm_param['target_columns_metadata'], \
                    lgbm_param['multi_output_estimator_dict']
                lgbm_param['Booster'] = lgbm_param.pop('booster')
                self._learner._le = params['le']
                self._learner.set_params(**lgbm_param)
            self._target_columns_metadata = params['target_columns_metadata']
            self._attribute_columns_names = params['attribute_columns_names']
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

        # if not d3m_utils.is_numeric(column_metadata['structural_type']):
        #     return False
        #
        return 'https://metadata.datadrivendiscovery.org/types/Attribute' in column_metadata.get('semantic_types',
                                                                                                 [])

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
