import os
from collections import OrderedDict
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple

import numpy as np  # type: ignore
import xgboost as xgb  # type: ignore
from sklearn.multioutput import MultiOutputRegressor  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult, SamplingCompositionalityMixin, ContinueFitMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import common_primitives

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    estimators: Optional[Union[xgb.XGBRegressor, List[xgb.XGBRegressor]]]
    booster: Optional[Union[xgb.Booster, List[xgb.Booster]]]
    objective: Optional[str]
    multi_output_estimator_dict: Optional[Dict]
    target_columns_metadata: Optional[List[OrderedDict]]


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
    importance_type = hyperparams.Enumeration[str](
        values=['gain', 'weight', 'cover', 'total_gain', 'total_cover'],
        default='gain',
        description='The feature importance type',
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


class XGBoostGBTreeRegressorPrimitive(SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                      ContinueFitMixin[Inputs, Outputs, Params, Hyperparams],
                                      SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A XGBoost classifier using ``xgb.XGBoostRegressor`` with GBTree Boosting type.

    It uses semantic types to determine which columns to operate on.
    """
    __author__ = 'TAMU DARPA D3M Team, TsungLin Yang <lin.yang@tamu.edu>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'cdbb80e4-e9de-4caa-a710-16b5d727b959',
            'version': '0.1.0',
            'name': "XGBoost GBTree regressor",
            'python_path': 'd3m.primitives.regression.xgboost_gbtree.Common',
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
            'primitive_family': metadata_base.PrimitiveFamily.REGRESSION,
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
        self._learner: Union[xgb.XGBRegressor, MultiOutputRegressor] = None
        self._target_columns_metadata: List[OrderedDict] = None
        # A dictionary recording estimator-target_column mapping.
        self._multi_output_estimator_dict: Dict = {}

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._new_training_data = True

    def _create_learner(self) -> None:
        self._learner = xgb.XGBRegressor(
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
            importance_type=self.hyperparams['importance_type'],
            base_score=self.hyperparams['base_score'],
            n_jobs=-1 if self.hyperparams['n_jobs'] is None else self.hyperparams['n_jobs'],
            random_state=self.random_seed,
            booster='gbtree',
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

    def _store_target_columns_metadata(self, outputs: Outputs) -> None:
        self._target_columns_metadata = self._get_target_columns_metadata(outputs.metadata)

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
            self._learner = MultiOutputRegressor(self._learner)

        # convert to np.array in order to unify the feature name based on xgboost's implementation
        if type(inputs) is not np.array:
            inputs = np.array(inputs)
            fit_outputs = np.array(fit_outputs)

        self._learner.fit(inputs, fit_outputs)

        if isinstance(self._learner, MultiOutputRegressor):
            for index, estimator in enumerate(self._learner.estimators_):
                self._multi_output_estimator_dict[outputs.columns.values[index]] = index

        self._store_target_columns_metadata(outputs)

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
        def continue_xgb_booster(xgb_model: xgb.XGBRegressor, inputs: Inputs,
                                 output_values: Union[np.ndarray, Outputs], num_of_boosting_round: int) -> None:
            dtrain = xgb.DMatrix(data=inputs.values, label=output_values)
            model_param = xgb_model.get_xgb_params()
            del (model_param['n_estimators'])
            model_param['objective'] = xgb_model.objective
            booster = xgb.train(params=model_param, dtrain=dtrain,
                                num_boost_round=num_of_boosting_round,
                                xgb_model=xgb_model.get_booster())
            xgb_model.set_params(_Booster=booster)

        # A special case for sklearn. It prefers an 1D array instead of 2D when there is only one target.
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            continue_xgb_booster(self._learner, inputs, np.ravel(outputs), self.hyperparams['n_more_estimators'])
        else:
            for _, output in outputs.iteritems():
                estimator_index = self._multi_output_estimator_dict.get(output.name, None)
                if estimator_index is None:
                    raise exceptions.InvalidArgumentValueError(
                        'Unseen target column when continuing fit {}'.format(output.name))
                estimator = self._learner.estimators_[self._multi_output_estimator_dict[output.name]]
                continue_xgb_booster(estimator, inputs, output, self.hyperparams['n_more_estimators'])

        self._store_target_columns_metadata(outputs)

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
        return outputs

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        selected_inputs, columns_to_use = self._select_inputs_columns(inputs)

        predictions = self._learner.predict(selected_inputs.values)

        output_columns = [self._wrap_predictions(predictions)]

        outputs = base_utils.combine_columns(
            inputs, columns_to_use, output_columns,
            return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'],
        )

        return CallResult(outputs)

    def produce_feature_importances(self, *, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._learner:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        # TODO: is feature importances the same for every target?
        if hasattr(self._learner, 'feature_importances_'):
            feature_importances_array = self._learner.feature_importances_
        else:
            feature_importances_array = self._learner.estimators_[0].feature_importances_

        if feature_importances_array.ndim == 1:
            feature_importances_array = feature_importances_array.reshape((1, feature_importances_array.shape[0]))

        return CallResult(container.DataFrame(feature_importances_array, generate_metadata=True))

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

    def get_params(self) -> Params:
        if not self._learner:
            return Params(
                estimators=None,
                objective=None,
                booster=None,
                multi_output_estimator_dict=None,
                target_columns_metadata=None,
            )

        return Params(
            estimators=self._learner.estimators_ if isinstance(self._learner, MultiOutputRegressor) else self._learner,
            objective=self._learner.objective
            if not isinstance(self._learner, MultiOutputRegressor) else self._learner.estimators_[0].objective,
            booster=self._learner.get_booster() if not isinstance(self._learner, MultiOutputRegressor) else [
                estimator.get_booster() for estimator in self._learner.estimators_],
            multi_output_estimator_dict=self._multi_output_estimator_dict
            if isinstance(self._learner, MultiOutputRegressor) else {},
            target_columns_metadata=self._target_columns_metadata,
        )

    def set_params(self, *, params: Params) -> None:
        if not all(params[param] is not None for param in
                   ['estimators', 'booster', 'objective', 'multi_output_estimator_dict', 'target_columns_metadata']):
            self._learner = None
        else:
            if isinstance(self._learner, MultiOutputRegressor):
                self._learner.estimators_ = params['estimators']
                self._multi_output_estimator_dict = params['multi_output_estimator_dict']
            else:
                self._create_learner()
                self._learner.set_params(_Booster=params['booster'], objective=params['objective'])
            self._target_columns_metadata = params['target_columns_metadata']

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
