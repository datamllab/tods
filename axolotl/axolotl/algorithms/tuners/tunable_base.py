import logging
import multiprocessing

import os
import uuid
import copy
from typing import Tuple
import re
import numpy as np

from d3m.metadata import hyperparams
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline

from kerastuner.engine import trial as trial_module

from axolotl import predefined_pipelines
from axolotl.algorithms.tuners import custom_hps
from axolotl.algorithms.base import PipelineSearchBase
from axolotl.algorithms.dummy import dummy_ranking_function
from axolotl.algorithms.tuners.hyperparameters import HyperParameters, PIPELINE_CHOICE
from axolotl.utils import schemas as schemas_utils

logger = logging.getLogger(__name__)


class TunableBase(PipelineSearchBase):

    def __init__(self, problem_description, backend,
                 primitives_blocklist=None, ranking_function=None, num_eval_trials=None):
        if ranking_function is None:
            ranking_function = dummy_ranking_function
        if num_eval_trials is None:
            num_eval_trials = multiprocessing.cpu_count()
        super(TunableBase, self).__init__(problem_description, backend,
                                          primitives_blocklist=primitives_blocklist, ranking_function=ranking_function)
        # TODO update this to be defined on problem/metrics terms
        self.data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
        self.data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']

        self.scoring_pipeline = schemas_utils.get_scoring_pipeline()
        self.scoring_params = None

        self.metrics = problem_description['problem']['performance_metrics']

        self.oracle = None
        self.tuner_id = 'tuner'
        self.hyperparameters = HyperParameters()
        self.pipeline_candidates = {}
        self.num_eval_trials = num_eval_trials

    def set_pipeline_candidates(self, input_data, pipeline_candidates):
        if pipeline_candidates is None:
            problem = self.problem_description
            # ToDo should use fetch(input_data, problem, schemas_utils.PIPELINES_DB_DIR)
            for pipeline in predefined_pipelines.fetch_from_file(problem, schemas_utils.PIPELINES_DB_DIR):
                self.pipeline_candidates[pipeline.id] = pipeline
        elif isinstance(pipeline_candidates, list):
            for pipeline in pipeline_candidates:
                self.pipeline_candidates[pipeline.id] = pipeline
        elif isinstance(pipeline_candidates, dict):
            self.pipeline_candidates = pipeline_candidates
        else:
            raise ValueError('pipeline_candidate should be None, list or dict')

    def init_search_space(self):
        pipeline_id = hyperparams.Enumeration[str](
            values=list(self.pipeline_candidates.keys()),
            default=list(self.pipeline_candidates.keys())[0],
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )
        self.hyperparameters.retrieve(PIPELINE_CHOICE, pipeline_id)
        for pipeline in self.pipeline_candidates.values():
            self._get_pipeline_search_space(pipeline)

    def _get_pipeline_search_space(self, pipeline):
        PREFIX_STEP = 'step'
        with self.hyperparameters.conditional_scope(PIPELINE_CHOICE, pipeline.id):
            for i, step in enumerate(pipeline.steps):
                with self.hyperparameters.name_scope('{}{}'.format(PREFIX_STEP, i)):
                    primitive = step.primitive
                    self._get_primitive_search_space(primitive)

    def _get_primitive_search_space(self, primitive):
        hyperparameters = primitive.metadata.query()['primitive_code']['hyperparams']
        primitive_python_path = primitive.metadata.query()['python_path']
        name = primitive_python_path
        config = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].configuration
        custom_config = custom_hps.config.get(primitive_python_path, None)
        if not custom_config is None:
            config._dict.update(custom_config)
        with self.hyperparameters.name_scope(name):
            for param_name, param_info in hyperparameters.items():
                if self.is_tunable(param_info['semantic_types']):
                    param_val = config[param_name]
                    # SortedSet.to_simple_structure() has bug, so we skip it.
                    if isinstance(param_val, (hyperparams.List, hyperparams.Set)):
                        continue
                    self.hyperparameters.retrieve(param_name, param_val)
                    if isinstance(param_val, hyperparams.Choice):
                        for choice_name, choice_val in param_val.choices.items():
                            with self.hyperparameters.conditional_scope(param_name, choice_name):
                                for sub_param_name, sub_param_val in choice_val.configuration.items():
                                    if sub_param_name != 'choice':
                                        self.hyperparameters.retrieve(sub_param_name, sub_param_val)

    def is_tunable(self, semantic_types: Tuple[str, ...]) -> bool:
        return any('tuning' in t.lower() for t in semantic_types)

    def search_fit(self, input_data, time_limit=300, *, expose_values=False, pipeline_candidates=None):
        self.set_pipeline_candidates(input_data, pipeline_candidates)
        self.init_search_space()
        return super(TunableBase, self).search_fit(input_data, time_limit, expose_values=expose_values)

    def _search(self, time_left):
        trials = self.create_trials(num_trials=self.num_eval_trials)
        if len(trials) == 0:
            logger.info('Oracle trigger exit')
            return []
        results = self.run_trials(trials, input_data=self.input_data)
        self.end_trials(trials)
        return results

    def run_trials(self, trials, **fit_kwargs):
        pipelines = []
        id_2_trials = {}

        for trial in trials:
            hp = trial.hyperparameters
            try:
                pipeline = self.build_pipeline(hp)
                id_2_trials[pipeline.id] = trial
                pipelines.append(pipeline)
            except Exception as e:
                logger.error('Current trial is failed. Error: {}'.format(e))
                trial.status = trial_module.TrialStatus.INVALID

        input_data = fit_kwargs.pop('input_data')

        pipeline_results = self.backend.evaluate_pipelines(
            problem_description=self.problem_description,
            pipelines=pipelines,
            input_data=input_data,
            metrics=self.metrics,
            data_preparation_pipeline=self.data_preparation_pipeline,
            scoring_pipeline=self.scoring_pipeline,
            data_preparation_params=self.data_preparation_params,
        )

        results = []
        for result in pipeline_results:
            trial = id_2_trials[result.pipeline.id]
            if result.status == 'ERRORED':
                logger.error('Current trial is failed. Error: {}'.format(result.error))
                trial.status = trial_module.TrialStatus.INVALID
            else:
                scores = result.scores
                # scores = runtime_module.combine_folds(scores)
                summarize_performance = schemas_utils.summarize_performance_metrics(scores)
                metrics = self._get_pipeline_metrics(summarize_performance)
                self.oracle.update_trial(
                    trial.trial_id, metrics=metrics
                )
                trial.status = trial_module.TrialStatus.COMPLETED
            results.append(self.ranking_function(result))
        return results

    def build_pipeline(self, hyperparameters):
        """
            hyperparameters example:
                {
                    'STEP5/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/max_percent_null: 0,
                    'STEP7/d3m.primitives.data_preprocessing.robust_scaler.SKlearn/quantile_range: (2.798121390864261, 14.852664215409096),
                }
        """
        values = hyperparameters.values
        pipeline_id = hyperparameters.get_pipeline_id()
        pipeline = copy.deepcopy(self.pipeline_candidates[pipeline_id])
        pipeline.id = str(uuid.uuid4())
        # update time
        pipeline.created = Pipeline().created

        skip_hps = set()
        # for key in sorted(values.keys()):
        for hp in hyperparameters.space:
            if hyperparameters.is_active(hp) and hp.name not in skip_hps and hp.name != PIPELINE_CHOICE:
                key = hp.name
                step, primitive_name, hp_name = hyperparameters.get_name_parts(key)
                value = values[key]
                step_idx = self.__get_step_idx_by_name(step)
                if step_idx is None:
                    raise KeyError('{} not in the pipeline'.format(primitive_name))
                primitive_step = pipeline.steps[step_idx]
                arg_type = ArgumentType.VALUE
                # In order to avoid the following error
                # Value '0' for hyper-parameter \
                # 'STEP8/d3m.primitives.classification.xgboost_gbtree.DataFrameCommon/max_delta_step' \
                # is not an instance of the structural type: typing.Union[int, NoneType]
                # Here is workaround
                if isinstance(value, np.int64):
                    value = int(value)
                elif isinstance(value, np.str_):
                    value = str(value)
                elif isinstance(value, np.bool_):
                    value = bool(value)
                if hp_name in primitive_step.hyperparams:
                    del primitive_step.hyperparams[hp_name]
                # Handle Choice
                if isinstance(hp, hyperparams.Choice):
                    choice_cls = hp.choices[value]
                    _vals = {}
                    for name in choice_cls.configuration:
                        if name == 'choice':
                            _vals[name] = value
                        else:
                            _key = os.path.join(step, primitive_name, name)
                            _vals[name] = values[_key]
                            skip_hps.add(_key)
                    value = choice_cls(_vals)
                primitive_step.add_hyperparameter(name=hp_name, argument_type=arg_type,
                                                  data=value)
        return pipeline

    def __get_step_idx_by_name(self, prefix_primitive_name):
        regex = r"(?<=STEP)\d+"
        match = re.search(regex, prefix_primitive_name, re.IGNORECASE)
        if match:
            return int(match.group(0))
        return None

    def _get_pipeline_metrics(self, summarize_performance):
        metrics = {}
        for name, info in summarize_performance.items():
            metrics[name] = info['mean']
        return metrics

    def end_trials(self, trials):
        """A hook called after each trial is run.

        # Arguments:
            trial: A `Trial` instance.
        """
        [self.oracle.end_trial(trial.trial_id, trial.status) for trial in trials]
        # self.oracle.update_space(trial.hyperparameters)

    def create_trials(self, num_trials):
        trials = []
        for i in range(num_trials):
            try:
                trial = self.oracle.create_trial('{}_{}'.format(self.tuner_id, i))
            except:
               break

            if trial.status == trial_module.TrialStatus.STOPPED:
                break
            else:
                trials.append(trial)
        return trials
