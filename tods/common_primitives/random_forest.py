import os
from collections import OrderedDict
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sklearn.tree  # type: ignore
from sklearn.ensemble.forest import RandomForestClassifier  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult, ProbabilisticCompositionalityMixin, SamplingCompositionalityMixin, ContinueFitMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import common_primitives


Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    estimators: Optional[List[sklearn.tree.DecisionTreeClassifier]]
    classes: Optional[Union[np.ndarray, List[np.ndarray]]]
    n_classes: Optional[Union[int, List[int]]]
    n_features: Optional[int]
    n_outputs: Optional[int]
    attribute_columns_names: Optional[List[str]]
    target_columns_metadata: Optional[List[OrderedDict]]
    target_columns_names: Optional[List[str]]
    oob_score: Optional[float]
    oob_decision_function: Optional[Union[np.ndarray, List[np.ndarray]]]


class Hyperparams(hyperparams.Hyperparams):
    # TODO: How to define it better?
    #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/150
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
    criterion = hyperparams.Enumeration[str](
        values=['gini', 'entropy'],
        default='gini',
        description='The function to measure the quality of a split.'
                    ' Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.'
                    ' Note: this parameter is tree-specific.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    max_features = hyperparams.Union[Union[int, float, str, None]](
        configuration=OrderedDict(
            # TODO: How to mark it as depending on the number of input features?
            fixed=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=1,
                description='Consider "max_features" features at each split.'),
            ratio=hyperparams.Uniform(
                lower=0,
                upper=1,
                default=0.25,
                lower_inclusive=True,
                # "ratio" == 1.0 is equal to "all_features", we do not want to have it twice.
                # Moreover, this makes it possible to differentiate between "fixed" and "ratio" just by the value.
                upper_inclusive=False,
                description='A percentage. "int(max_features * n_features)" features are considered at each split.',
            ),
            calculated=hyperparams.Enumeration[str](
                values=['sqrt', 'log2'],
                default='sqrt',
                description='If "sqrt", then "max_features = sqrt(n_features)". If "log2", then "max_features = log2(n_features)".',
            ),
            all_features=hyperparams.Constant(
                default=None,
                description='"max_features = n_features".',
            ),
        ),
        default='calculated',
        description='The number of features to consider when looking for the best split.'
                    ' The search for a split does not stop until at least one valid partition of the node samples is found,'
                    ' even if it requires to effectively inspect more than "max_features" features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    max_depth = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=10,
            ),
            unlimited=hyperparams.Constant(
                default=None,
                description='Nodes are expanded until all leaves are pure or until all leaves contain less than "min_samples_split" samples.',
            ),
        ),
        default='unlimited',
        description='The maximum depth of the tree.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_samples_split = hyperparams.Union[Union[int, float]](
        configuration=OrderedDict(
            # TODO: How to mark it as depending on the number of input samples?
            fixed=hyperparams.Bounded[int](
                lower=2,
                upper=None,
                default=2,
                description='Consider "min_samples_split" as the minimum number.',
            ),
            ratio=hyperparams.Uniform(
                lower=0,
                upper=1,
                default=0.25,
                lower_inclusive=False,
                upper_inclusive=True,
                description='A percentage. "ceil(min_samples_split * n_samples)" are the minimum number of samples for each split.',
            ),
        ),
        default='fixed',
        description='The minimum number of samples required to split an internal node.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_samples_leaf = hyperparams.Union[Union[int, float]](
        configuration=OrderedDict(
            # TODO: How to mark it as depending on the number of input samples?
            fixed=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=1,
                description='Consider "min_samples_leaf" as the minimum number.',
            ),
            ratio=hyperparams.Uniform(
                lower=0,
                upper=0.5,
                default=0.25,
                lower_inclusive=False,
                upper_inclusive=True,
                description='A percentage. "ceil(min_samples_leaf * n_samples)" are the minimum number of samples for each node.',
            ),
        ),
        default='fixed',
        description='The minimum number of samples required to be at a leaf node.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_weight_fraction_leaf = hyperparams.Uniform(
        lower=0,
        upper=0.5,
        default=0,
        upper_inclusive=True,
        description='The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    max_leaf_nodes = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=2,
                upper=None,
                default=10,
            ),
            unlimited=hyperparams.Constant(
                default=None,
                description='Unlimited number of leaf nodes.',
            ),
        ),
        default='unlimited',
        description='Grow trees with "max_leaf_nodes" in best-first fashion. Best nodes are defined as relative reduction in impurity.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    min_impurity_decrease = hyperparams.Bounded[float](
        lower=0.0,
        upper=None,
        default=0.0,
        description='A node will be split if this split induces a decrease of the impurity greater than or equal to this value.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    bootstrap = hyperparams.Enumeration[str](
        values=['bootstrap', 'bootstrap_with_oob_score', 'disabled'],
        default='bootstrap',
        description='Whether bootstrap samples are used when building trees.'
                    ' And whether to use out-of-bag samples to estimate the generalization accuracy.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    # In reality values could also be -2 and so on, which would mean all CPUs minus 1,
    # but this does not really seem so useful here, so it is not exposed.
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
    error_on_no_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no column is selected/provided. Otherwise issue a warning.",
    )


# TODO: Support weights on samples.
#       There is a "https://metadata.datadrivendiscovery.org/types/InstanceWeight" semantic type which should be used for this.
#       See: https://gitlab.com/datadrivendiscovery/d3m/issues/151
# TODO: How to use/determine class weights?
class RandomForestClassifierPrimitive(ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                      SamplingCompositionalityMixin[Inputs, Outputs, Params, Hyperparams],
                                      ContinueFitMixin[Inputs, Outputs, Params, Hyperparams],
                                      SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A random forest classifier using ``sklearn.ensemble.forest.RandomForestClassifier``.

    It uses semantic types to determine which columns to operate on.
    """

    __author__ = 'Oxford DARPA D3M Team, Rob Zinkov <zinkov@robots.ox.ac.uk>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '37c2b19d-bdab-4a30-ba08-6be49edcc6af',
            'version': '0.4.0',
            'name': "Random forest classifier",
            'python_path': 'd3m.primitives.classification.random_forest.Common',
            'keywords': ['random forest', 'decision tree'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:zinkov@robots.ox.ac.uk',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/random_forest.py',
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
                metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
            'hyperparams_to_tune': [
                'max_leaf_nodes',
                'criterion',
                'max_features',
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
        self._learner: RandomForestClassifier = None
        self._attribute_columns_names: List[str] = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._target_columns_names: List[str] = None

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._new_training_data = True

    def _create_learner(self) -> None:
        self._learner = RandomForestClassifier(
            n_estimators=self.hyperparams['n_estimators'],
            criterion=self.hyperparams['criterion'],
            max_features=self.hyperparams['max_features'],
            max_depth=self.hyperparams['max_depth'],
            min_samples_split=self.hyperparams['min_samples_split'],
            min_samples_leaf=self.hyperparams['min_samples_leaf'],
            min_weight_fraction_leaf=self.hyperparams['min_weight_fraction_leaf'],
            max_leaf_nodes=self.hyperparams['max_leaf_nodes'],
            min_impurity_decrease=self.hyperparams['min_impurity_decrease'],
            bootstrap=self.hyperparams['bootstrap'] in ['bootstrap', 'bootstrap_with_oob_score'],
            oob_score=self.hyperparams['bootstrap'] in ['bootstrap_with_oob_score'],
            n_jobs=-1 if self.hyperparams['n_jobs'] is None else self.hyperparams['n_jobs'],
            warm_start=True,
            random_state=self._random_state,
            verbose=self._verbose,
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
            semantic_types = [semantic_type for semantic_type in semantic_types if semantic_type != 'https://metadata.datadrivendiscovery.org/types/TrueTarget']
            column_metadata['semantic_types'] = semantic_types

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    def _store_columns_metadata_and_names(self, inputs: Inputs, outputs: Outputs) -> None:
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

        self._store_columns_metadata_and_names(inputs, outputs)

        # We skip if there are no columns. If "error_on_no_columns" is set,
        # exception should have already been raised.
        if len(inputs.columns) and len(outputs.columns):
            self._learner.fit(inputs, fit_outputs)

            assert self._learner.n_features_ == len(self._attribute_columns_names), (self._learner.n_features_, len(self._attribute_columns_names))
            assert self._learner.n_outputs_ == len(self._target_columns_metadata), (self._learner.n_outputs_, len(self._target_columns_metadata))
            assert self._learner.n_outputs_ == len(self._target_columns_names), (self._learner.n_outputs_, len(self._target_columns_names))

        return CallResult(None)

    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None or self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        # This model is not improving fitting if called multiple times on the same data.
        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        if self._learner is None:
            self._create_learner()

        n_estimators = self._learner.get_params()['n_estimators']
        n_estimators += self.hyperparams['n_more_estimators']
        self._learner.set_params(n_estimators=n_estimators)

        inputs, _ = self._select_inputs_columns(self._training_inputs)
        outputs, _ = self._select_outputs_columns(self._training_outputs)

        # A special case for sklearn. It prefers an 1D array instead of 2D when there is only one target.
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            fit_outputs = np.ravel(outputs)
        else:
            fit_outputs = outputs

        self._store_columns_metadata_and_names(inputs, outputs)

        # We skip if there are no columns. If "error_on_no_columns" is set,
        # exception should have already been raised.
        if len(inputs.columns) and len(outputs.columns):
            self._learner.fit(inputs, fit_outputs)

            assert self._learner.n_features_ == len(self._attribute_columns_names), (self._learner.n_features_, len(self._attribute_columns_names))
            assert self._learner.n_outputs_ == len(self._target_columns_metadata), (self._learner.n_outputs_, len(self._target_columns_metadata))
            assert self._learner.n_outputs_ == len(self._target_columns_names), (self._learner.n_outputs_, len(self._target_columns_names))

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

    def _predictions_from_proba(self, proba: np.ndarray) -> np.ndarray:
        """
        This is copied from ``ForestClassifier.predict``, but also includes a bugfix for
        `this issue`_.

        .. _this issue: https://github.com/scikit-learn/scikit-learn/issues/11451
        """

        if self._learner.n_outputs_ == 1:
            return self._learner.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            predictions = []

            for k in range(self._learner.n_outputs_):
                predictions.append(self._learner.classes_[k].take(np.argmax(proba[k], axis=1), axis=0))

            return np.array(predictions).T

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._learner is None:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        selected_inputs, columns_to_use = self._select_inputs_columns(inputs)

        # We skip if there are no columns. If "error_on_no_columns" is set, exception should have already been raised.
        # The number of columns should match the number during fitting, so if columns are available now we assume that
        # the learner has been really fitted.
        output_columns: List[Outputs] = []
        if len(selected_inputs.columns):
            # We are not using "predict" directly because of a bug.
            # See: https://github.com/scikit-learn/scikit-learn/issues/11451
            proba = self._learner.predict_proba(selected_inputs)
            predictions = self._predictions_from_proba(proba)

            output_columns = [self._wrap_predictions(predictions)]

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return CallResult(outputs)

    def produce_feature_importances(self, *, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._learner is None:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        if len(getattr(self._learner, 'estimators_', [])):
            feature_importances_array = self._learner.feature_importances_.reshape((1, len(self._attribute_columns_names)))

            feature_importances = container.DataFrame(feature_importances_array, generate_metadata=True)
            feature_importances.columns = self._attribute_columns_names
            for k in range(len(self._attribute_columns_names)):
                feature_importances.metadata = feature_importances.metadata.update_column(k, {'name': self._attribute_columns_names[k]})

        else:
            feature_importances = container.DataFrame(generate_metadata=True)

        return CallResult(feature_importances)

    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> CallResult[Sequence[Outputs]]:
        if self._learner is None:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        inputs, _ = self._select_inputs_columns(inputs)

        # We skip if there are no columns. If "error_on_no_columns" is set, exception should have already been raised.
        # The number of columns should match the number during fitting, so if columns are available now we assume that
        # the learner has been really fitted.
        samples = []
        if len(inputs.columns):
            for i in range(num_samples):
                proba = self._random_state.choice(self._learner.estimators_).predict_proba(inputs)
                predictions = self._predictions_from_proba(proba)
                samples.append(self._wrap_predictions(predictions))

        return CallResult(samples)

    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._learner is None:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        inputs, _ = self._select_inputs_columns(inputs)
        outputs, _ = self._select_outputs_columns(outputs)

        # We skip if there are no columns. If "error_on_no_columns" is set, exception should have already been raised.
        # The number of columns should match the number during fitting, so if columns are available now we assume that
        # the learner has been really fitted.
        if len(inputs.columns) and len(outputs.columns):
            if outputs.shape[1] != self._learner.n_outputs_:
                raise exceptions.InvalidArgumentValueError("\"outputs\" argument does not have the correct number of target columns.")

            log_proba = self._learner.predict_log_proba(inputs)

            # Making it always a list, even when only one target.
            if self._learner.n_outputs_ == 1:
                log_proba = [log_proba]
                classes = [self._learner.classes_]
            else:
                classes = self._learner.classes_

            samples_length = inputs.shape[0]

            log_likelihoods = []
            for k in range(self._learner.n_outputs_):
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

            # TODO: Copy any other metadata?
            for k in range(self._learner.n_outputs_):
                column_metadata = outputs.metadata.query_column(k)
                if 'name' in column_metadata:
                    results.metadata = results.metadata.update_column(k, {'name': column_metadata['name']})

        else:
            results = container.DataFrame(generate_metadata=True)

        return CallResult(results)

    def get_params(self) -> Params:
        if self._learner is None:
            return Params(
                estimators=None,
                classes=None,
                n_classes=None,
                n_features=None,
                n_outputs=None,
                attribute_columns_names=None,
                target_columns_metadata=None,
                target_columns_names=None,
                oob_score=None,
                oob_decision_function=None,
            )

        elif not len(getattr(self._learner, 'estimators_', [])):
            return Params(
                estimators=[],
                classes=None,
                n_classes=None,
                n_features=None,
                n_outputs=None,
                attribute_columns_names=self._attribute_columns_names,
                target_columns_metadata=self._target_columns_metadata,
                target_columns_names=self._target_columns_names,
                oob_score=None,
                oob_decision_function=None,
            )

        return Params(
            estimators=self._learner.estimators_,
            classes=self._learner.classes_,
            n_classes=self._learner.n_classes_,
            n_features=self._learner.n_features_,
            n_outputs=self._learner.n_outputs_,
            attribute_columns_names=self._attribute_columns_names,
            target_columns_metadata=self._target_columns_metadata,
            target_columns_names=self._target_columns_names,
            oob_score=getattr(self._learner, 'oob_score_', None),
            oob_decision_function=getattr(self._learner, 'oob_decision_function_', None),
        )

    def set_params(self, *, params: Params) -> None:
        if params['estimators'] is None:
            self._learner = None
        else:
            self._create_learner()

            if params['estimators']:
                self._learner.estimators_ = params['estimators']
            if params['classes'] is not None:
                self._learner.classes_ = params['classes']
            if params['n_classes'] is not None:
                self._learner.n_classes_ = params['n_classes']
            if params['n_features'] is not None:
                self._learner.n_features_ = params['n_features']
            if params['n_outputs'] is not None:
                self._learner.n_outputs_ = params['n_outputs']
            self._attribute_columns_names = params['attribute_columns_names']
            self._target_columns_metadata = params['target_columns_metadata']
            self._target_columns_names = params['target_columns_names']
            if params['oob_score'] is not None:
                self._learner.oob_score_ = params['oob_score']
            if params['oob_decision_function'] is not None:
                self._learner.oob_decision_function_ = params['oob_decision_function']

            if getattr(self._learner, 'estimators_', []):
                # When continuing fitting, we are increasing "n_estimators", so we have to make sure
                # "n_estimators" matches the number of fitted estimators which might be different
                # from initial value set from through the hyper-parameter.
                self._learner.set_params(n_estimators=len(self._learner.estimators_))

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

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_inputs_columns'], self.hyperparams['exclude_inputs_columns'], can_use_column)

        if not columns_to_use:
            if self.hyperparams['error_on_no_columns']:
                raise ValueError("No inputs columns.")
            else:
                self.logger.warning("No inputs columns.")

        if self.hyperparams['use_inputs_columns'] and columns_to_use and columns_not_to_use:
            self.logger.warning("Not all specified inputs columns can be used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _can_use_outputs_column(self, outputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        return 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in column_metadata.get('semantic_types', [])

    def _get_outputs_columns(self, outputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_outputs_column(outputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(outputs_metadata, self.hyperparams['use_outputs_columns'], self.hyperparams['exclude_outputs_columns'], can_use_column)

        if not columns_to_use:
            if self.hyperparams['error_on_no_columns']:
                raise ValueError("No outputs columns.")
            else:
                self.logger.warning("No outputs columns.")

        if self.hyperparams['use_outputs_columns'] and columns_to_use and columns_not_to_use:
            self.logger.warning("Not all specified outputs columns can be used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _select_inputs_columns(self, inputs: Inputs) -> Tuple[Inputs, List[int]]:
        columns_to_use = self._get_inputs_columns(inputs.metadata)

        return inputs.select_columns(columns_to_use, allow_empty_columns=True), columns_to_use

    def _select_outputs_columns(self, outputs: Outputs) -> Tuple[Outputs, List[int]]:
        columns_to_use = self._get_outputs_columns(outputs.metadata)

        return outputs.select_columns(columns_to_use, allow_empty_columns=True), columns_to_use
