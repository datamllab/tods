import argparse
import collections
import copy
import datetime
import enum
import json
import logging
import os.path
import re
import sys
import traceback
import typing
import uuid
import yaml

import dateparser  # type: ignore
import git  # type: ignore
import GPUtil  # type: ignore

import d3m
from d3m import container, environment_variables, exceptions, utils, types
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.primitive_interfaces import base

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import lib as pyarrow_lib  # type: ignore
except ModuleNotFoundError:
    pyarrow_lib = None

__all__ = ('PipelineRun', 'User', 'RuntimeEnvironment')

logger = logging.getLogger(__name__)

DOCKER_MAC_ADDRESS_MASK = 0x0242ac110000
PROC_INFO_RE = re.compile(r'^([^:]+?)\s*:\s*(.*)$')
PROC_MEMORY_PATH = '/proc/meminfo'
PROC_CPU_PATH = '/proc/cpuinfo'
PROC_CPU_MODEL_NAME_KEY = 'model name'
PROC_CPU_PHYSICAL_ID_KEY = 'physical id'
PROC_CPU_CORES_KEY = 'cpu cores'
PROC_TOTAL_MEMORY_KEY = 'MemTotal'
CGROUP_MEMORY_LIMIT_PATH = '/sys/fs/cgroup/memory/memory.limit_in_bytes'
CGROUP_CPU_SHARES_PATH = '/sys/fs/cgroup/cpu/cpu.shares'
CGROUP_CPU_CFS_PERIOD_US_PATH = '/sys/fs/cgroup/cpu/cpu.cfs_period_us'
CGROUP_CPU_CFS_QUOTA_US_PATH = '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'

WORKER_ID_NAMESPACE = uuid.UUID('2e4b9ab7-2207-4975-892b-0e01bf95babf')

# Comma because we unpack the list of validators returned from "load_schema_validators".
PIPELINE_RUN_SCHEMA_VALIDATOR, = utils.load_schema_validators(metadata_base.SCHEMAS, ('pipeline_run.json',))

PIPELINE_RUN_SCHEMA_VERSION = 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline_run.json'


class User(dict):
    def __init__(self, id_: str, chosen: bool = False, rationale: str = None) -> None:
        super().__init__()

        self['id'] = id_
        self['chosen'] = chosen

        if rationale is not None:
            self['rationale'] = rationale

    @classmethod
    def _yaml_representer(cls, dumper: yaml.Dumper, data: typing.Any) -> typing.Any:
        return dumper.represent_dict(data)


utils.yaml_add_representer(User, User._yaml_representer)


class PipelineRunStep:
    def __init__(
        self, step_type: metadata_base.PipelineStepType, start: str, environment: typing.Dict[str, typing.Any] = None
    ) -> None:
        self.type = step_type
        self.status: typing.Dict[str, typing.Any] = {}
        self.start: str = start
        self.end: str = None
        self.environment = environment

    def to_json_structure(self) -> typing.Dict:
        if self.start is None:
            raise exceptions.InvalidStateError("Start timestamp not set.")

        if self.end is None:
            raise exceptions.InvalidStateError("End timestamp not set.")

        if 'state' not in self.status:
            raise exceptions.InvalidStateError("Status not set.")

        json_structure = {
            'type': self.type.name,
            'status': self.status,
            'start': self.start,
            'end': self.end
        }

        if self.environment is not None:
            json_structure['environment'] = self.environment

        return json_structure

    def set_successful(self, message: str = None) -> None:
        self.status['state'] = metadata_base.PipelineRunStatusState.SUCCESS.name
        if message is not None and message:
            self.status['message'] = message

    def set_failed(self, message: str = None) -> None:
        self.status['state'] = metadata_base.PipelineRunStatusState.FAILURE.name
        if message is not None and message:
            self.status['message'] = message

    def set_end_timestamp(self) -> None:
        self.end = utils.datetime_for_json(datetime.datetime.now(datetime.timezone.utc))


class PipelineRunPrimitiveStep(PipelineRunStep):
    def __init__(
        self, step: pipeline_module.PrimitiveStep, start: str, environment: typing.Dict[str, typing.Any] = None,
    ) -> None:
        super().__init__(
            step_type=metadata_base.PipelineStepType.PRIMITIVE,
            start=start,
            environment=environment
        )

        self.hyperparams: hyperparams_module.Hyperparams = None
        self.pipeline_hyperparams: typing.Set[str] = None
        self.random_seed: typing.Optional[int] = None
        self.method_calls: typing.List[typing.Dict[str, typing.Any]] = []
        self.arguments = step.arguments

    def to_json_structure(self) -> typing.Dict:
        json_structure = super().to_json_structure()

        # Validate that the Method calls are finished, and they have status.
        for method_call in self.method_calls:
            if 'end' not in method_call:
                raise exceptions.InvalidStateError("End timestamp not set.")
            if 'status' not in method_call:
                raise exceptions.InvalidStateError("Status not set.")

        if self.method_calls:
            json_structure['method_calls'] = self.method_calls

        if self.random_seed is not None:
            json_structure['random_seed'] = self.random_seed

        hyperparams_json_structure = self._hyperparams_to_json_structure()
        if hyperparams_json_structure is not None:
            json_structure['hyperparams'] = hyperparams_json_structure

        return json_structure

    def _hyperparams_to_json_structure(self) -> typing.Optional[typing.Dict]:
        if self.hyperparams is None:
            return None

        hyperparams_json = {}

        for hyperparameter_name, value in self.hyperparams.items():
            if hyperparameter_name in self.pipeline_hyperparams:
                continue

            hyperparams_json[hyperparameter_name] = {
                'type': metadata_base.ArgumentType.VALUE.name,
                'data': self.hyperparams.configuration[hyperparameter_name].value_to_json_structure(value),
            }

        if hyperparams_json:
            return hyperparams_json
        else:
            return None

    def add_method_call(
        self, method_name: str, *, runtime_arguments: typing.Dict = None,
        environment: typing.Dict[str, typing.Any] = None
    ) -> int:
        """
        Returns
        -------
        The id of the method call.
        """

        if runtime_arguments is None:
            runtime_arguments = {}
        else:
            # We convert everything directly to json structure.
            def recurse(item: typing.Any) -> typing.Any:
                if isinstance(item, enum.Enum):
                    return item.name
                elif not isinstance(item, typing.Dict):
                    return item
                else:
                    _json_structure = {}
                    for key, value in item.items():
                        _json_structure[key] = recurse(value)
                    return _json_structure

            runtime_arguments = recurse(runtime_arguments)

        if method_name == '__init__' and runtime_arguments:
            raise exceptions.InvalidArgumentValueError(
                f'MethodCall with method `__init__` cannot have arguments. '
                f'Hyper-parameters are the arguments to `__init__`.'
            )

        method_call: typing.Dict[str, typing.Any] = {
            'name': method_name,
        }

        if runtime_arguments:
            method_call['arguments'] = runtime_arguments

        # we store everything as json structure.
        if environment is not None:
            method_call['environment'] = environment

        self.method_calls.append(method_call)
        return len(self.method_calls) - 1

    def set_method_call_start_timestamp(self, method_call_id: int) -> None:
        self.method_calls[method_call_id]['start'] = utils.datetime_for_json(datetime.datetime.now())

    def set_method_call_end_timestamp(self, method_call_id: int) -> None:
        if 'start' not in self.method_calls[method_call_id]:
            raise exceptions.InvalidStateError("Start timestamp not set.")
        self.method_calls[method_call_id]['end'] = utils.datetime_for_json(datetime.datetime.now())

    def set_method_call_result_metadata(self, method_call_id: int, result: typing.Union[base.CallResult, base.MultiCallResult]) -> None:
        metadata = None
        if isinstance(result, base.CallResult):
            if result.value is not None and isinstance(result.value, types.Container):
                metadata = {
                    # TODO: Should we use "to_internal_json_structure" here?
                    'value': result.value.metadata.to_json_structure()
                }
        elif isinstance(result, base.MultiCallResult):
            metadata = {
                # TODO: Should we use "to_internal_json_structure" here?
                produce_method_name: value.metadata.to_json_structure()
                for produce_method_name, value in result.values.items()
                if value is not None and isinstance(value, types.Container)
            }

        # check if metadata is empty
        if metadata is not None:
            for key, value in metadata.items():
                if value is not None:
                    self.method_calls[method_call_id]['metadata'] = metadata
                    break

    def set_method_call_successful(self, method_call_id: int, message: str = None) -> None:
        self.method_calls[method_call_id]['status'] = {
            'state': metadata_base.PipelineRunStatusState.SUCCESS.name,
        }
        if message is not None and message:
            self.method_calls[method_call_id]['status']['message'] = message

    def set_method_call_failed(self, method_call_id: int, message: str = None) -> None:
        self.method_calls[method_call_id]['status'] = {
            'state': metadata_base.PipelineRunStatusState.FAILURE.name,
        }
        if message is not None and message:
            self.method_calls[method_call_id]['status']['message'] = message

    def get_method_call_logging_callback(self, method_call_id: int) -> typing.Callable:
        if 'logging' not in self.method_calls[method_call_id]:
            self.method_calls[method_call_id]['logging'] = []
        return self.method_calls[method_call_id]['logging'].append


class PipelineRunSubpipelineStep(PipelineRunStep):
    def __init__(self, start: str, random_seed: int, environment: typing.Dict[str, typing.Any] = None) -> None:
        super().__init__(
            step_type=metadata_base.PipelineStepType.SUBPIPELINE,
            start=start,
            environment=environment,
        )

        self.random_seed = random_seed
        self.steps: typing.List[typing.Dict] = []

    def to_json_structure(self) -> typing.Dict:
        json_structure = super().to_json_structure()
        json_structure['random_seed'] = self.random_seed
        if self.steps:
            json_structure['steps'] = self.steps
        return json_structure

    def add_step(self, step: typing.Dict) -> None:
        self.steps.append(step)


class PipelineRun:
    STEPS = 'steps'
    METHOD_CALLS = 'method_calls'

    def __init__(
        self, pipeline: pipeline_module.Pipeline, problem_description: problem.Problem = None, *,
        phase: metadata_base.PipelineRunPhase, context: metadata_base.Context,
        environment: typing.Dict[str, typing.Any], random_seed: int, previous_pipeline_run: 'PipelineRun' = None,
        is_standard_pipeline: bool = False, users: typing.Sequence[User] = None,
    ) -> None:
        self.schema = PIPELINE_RUN_SCHEMA_VERSION

        self.pipeline = {
            'id': pipeline.id,
            'digest': pipeline.get_digest(),
        }

        self.datasets: typing.List[typing.Dict[str, typing.Any]] = []

        self.problem: typing.Dict[str, typing.Any] = None
        if problem_description is not None:
            self._set_problem(problem_description)

        self.steps: typing.List[PipelineRunStep] = []
        self.status: typing.Dict[str, typing.Any] = {}
        self.start: str = None
        self.end: str = None

        self.run: typing.Dict[str, typing.Any] = {
            'phase': phase.name,
            'is_standard_pipeline': is_standard_pipeline,
        }
        self.context = context
        self.previous_pipeline_run = previous_pipeline_run

        if users is None:
            self.users: typing.List[User] = []
        else:
            self.users = list(users)

        self.environment = environment
        self.random_seed = random_seed
        self.is_standard_pipeline = is_standard_pipeline

        self._components: typing.Dict[str, typing.Any] = {}
        self._step_start_timestamps: typing.Dict[int, str] = {}

    def _to_json_structure(self) -> typing.Dict:
        if self.start is None:
            raise exceptions.InvalidStateError("Start timestamp not set.")

        if self.end is None:
            raise exceptions.InvalidStateError("End timestamp not set.")

        if 'state' not in self.status:
            raise exceptions.InvalidStateError("Status not set.")

        # Scoring datasets are set only when scoring is used without data preparation.
        if 'scoring' in self.run:
            if 'data_preparation' in self.run:
                if 'datasets' in self.run['scoring']:
                    raise exceptions.InvalidStateError(
                        "Scoring datasets must not be provided when scoring is used with data preparation pipeline.",
                    )
            elif 'datasets' not in self.run['scoring']:
                raise exceptions.InvalidStateError(
                    "Scoring datasets must be provided when scoring is used without data preparation pipeline.",
                )

        json_structure = {
            'schema': self.schema,
            'pipeline': self.pipeline,
            'datasets': self.datasets,
            'status': self.status,
            'start': self.start,
            'end': self.end,
            'run': self.run,
            'environment': self.environment,
            'random_seed': self.random_seed,
        }

        if self.steps:
            json_structure['steps'] = [step.to_json_structure() for step in self.steps]

        if self.previous_pipeline_run is not None:
            json_structure['previous_pipeline_run'] = {
                'id': self.previous_pipeline_run.get_id()
            }

        if self.context is not None:
            json_structure['context'] = self.context.name

        if self.problem is not None:
            json_structure['problem'] = self.problem

        if self.users:
            json_structure['users'] = self.users

        json_structure['id'] = utils.compute_hash_id(json_structure)

        return json_structure

    def to_json_structure(self) -> typing.Dict:
        # We raise exception here instead of waiting for schema validation to fails to provide a more helpful error message.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/355
        if not self.is_standard_pipeline and not self.datasets:
            raise exceptions.InvalidStateError("Pipeline run for a non-standard pipeline cannot be converted to a JSON structure.")

        # TODO: Remove "utils.to_json_structure" once sure that "_to_json_structure" really returns a JSON structure.
        json_structure = utils.to_json_structure(self._to_json_structure())

        PIPELINE_RUN_SCHEMA_VALIDATOR.validate(json_structure)

        return json_structure

    def to_yaml(self, file: typing.IO[typing.Any], *, appending: bool = False, **kwargs: typing.Any) -> typing.Optional[str]:
        obj = self.to_json_structure()

        if appending and 'explicit_start' not in kwargs:
            kwargs['explicit_start'] = True

        return utils.yaml_dump(obj, stream=file, **kwargs)

    def add_input_dataset(self, dataset: container.Dataset) -> None:
        metadata = dataset.metadata.query(())
        self.datasets.append({
            'id': metadata['id'],
            'digest': metadata['digest'],
        })

    def add_primitive_step(self, step: pipeline_module.PrimitiveStep) -> int:
        if not isinstance(step, pipeline_module.PrimitiveStep):
            raise exceptions.InvalidArgumentTypeError('step must be of type PrimitiveStep, not {}'.format(type(step)))
        self.steps.append(
            PipelineRunPrimitiveStep(step, self._step_start_timestamps[len(self.steps)])
        )
        return len(self.steps) - 1

    def _get_primitive_step(self, primitive_step_id: int) -> PipelineRunPrimitiveStep:
        if primitive_step_id >= len(self.steps):
            raise exceptions.InvalidArgumentValueError('There does not exist a step with id {}'.format(primitive_step_id))

        primitive_step = self.steps[primitive_step_id]
        if not isinstance(primitive_step, PipelineRunPrimitiveStep):
            raise exceptions.InvalidArgumentValueError('Step id {} does not refer to a PipelineRunPrimitiveStep'.format(primitive_step_id))

        return primitive_step

    def set_primitive_step_hyperparams(
        self, primitive_step_id: int,
        hyperparams: hyperparams_module.Hyperparams,
        pipeline_hyperparams: typing.Dict[str, typing.Dict],
    ) -> None:
        primitive_step = self._get_primitive_step(primitive_step_id)
        primitive_step.hyperparams = hyperparams
        primitive_step.pipeline_hyperparams = set(pipeline_hyperparams.keys())

    def set_primitive_step_random_seed(self, primitive_step_id: int, random_seed: int) -> None:
        primitive_step = self._get_primitive_step(primitive_step_id)
        primitive_step.random_seed = random_seed

    def add_subpipeline_step(self, subpipeline_run: 'PipelineRun') -> int:
        pipeline_run_subpipeline_step = PipelineRunSubpipelineStep(
            self._step_start_timestamps[len(self.steps)], subpipeline_run.random_seed
        )

        for step_id, step in enumerate(subpipeline_run.steps):
            step_json = step.to_json_structure()
            pipeline_run_subpipeline_step.add_step(step_json)
            state = step_json['status']['state']
            message = step_json['status'].get('message', None)
            if state == metadata_base.PipelineRunStatusState.SUCCESS.name:
                pipeline_run_subpipeline_step.set_successful(message)
            elif state == metadata_base.PipelineRunStatusState.FAILURE.name:
                message = 'Failed on subpipeline step {}:\n{}'.format(step_id, message)
                pipeline_run_subpipeline_step.set_failed(message)
                if message is not None and message:
                    self.status['message'] = message
            else:
                raise exceptions.UnexpectedValueError('unknown subpipeline status state: {}'.format(state))

        self.steps.append(pipeline_run_subpipeline_step)

        return len(self.steps) - 1

    def add_method_call_to_primitive_step(
        self, primitive_step_id: int, method_name: str, *,
        runtime_arguments: typing.Dict = None, environment: typing.Dict[str, typing.Any] = None
    ) -> typing.Tuple[int, int]:
        if runtime_arguments is None:
            runtime_arguments = {}

        # TODO allow runtime arguments not specified in pipeline?
        primitive_step = self._get_primitive_step(primitive_step_id)
        method_call_id = primitive_step.add_method_call(
            method_name, runtime_arguments=runtime_arguments, environment=environment
        )
        return (primitive_step_id, method_call_id)

    def get_method_call_logging_callback(
        self, step_and_method_call_id: typing.Tuple[int, int]
    ) -> typing.Callable:
        step_id, method_call_id = step_and_method_call_id
        primitive_step = self._get_primitive_step(step_id)
        return primitive_step.get_method_call_logging_callback(method_call_id)

    def run_started(self) -> None:
        self.start = utils.datetime_for_json(datetime.datetime.now(datetime.timezone.utc))

    def _set_end_timestamp(self) -> None:
        self.end = utils.datetime_for_json(datetime.datetime.now(datetime.timezone.utc))

    def step_started(self, step_id: int) -> None:
        self._step_start_timestamps[step_id] = utils.datetime_for_json(datetime.datetime.now(datetime.timezone.utc))

    def method_call_started(self, step_and_method_call_id: typing.Tuple[int, int]) -> None:
        step_id, method_call_id = step_and_method_call_id
        primitive_step = self._get_primitive_step(step_id)
        primitive_step.set_method_call_start_timestamp(method_call_id)

    def set_method_call_result_metadata(
        self, step_and_method_call_id: typing.Tuple[int, int],
        result: typing.Union[base.CallResult, base.MultiCallResult]
    ) -> None:
        step_id, method_call_id = step_and_method_call_id
        primitive_step = self._get_primitive_step(step_id)
        primitive_step.set_method_call_result_metadata(method_call_id, result)

    def run_successful(self, message: str = None) -> None:
        self._set_end_timestamp()
        self.status['state'] = metadata_base.PipelineRunStatusState.SUCCESS.name
        if message is not None and message:
            self.status['message'] = message

    def step_successful(self, step_id: int, message: str = None) -> None:
        if step_id >= len(self.steps):
            raise exceptions.InvalidArgumentValueError('There does not exist a step with id {}'.format(step_id))
        self.steps[step_id].set_end_timestamp()
        self.steps[step_id].set_successful(message)

    def method_call_successful(self, step_and_method_call_id: typing.Tuple[int, int], message: str = None) -> None:
        step_id, method_call_id = step_and_method_call_id
        primitive_step = self._get_primitive_step(step_id)
        primitive_step.set_method_call_end_timestamp(method_call_id)
        primitive_step.set_method_call_successful(method_call_id, message)

    def run_failed(self, message: str = None) -> None:
        self._set_end_timestamp()
        self.status['state'] = metadata_base.PipelineRunStatusState.FAILURE.name
        if message is not None and message:
            self.status['message'] = message

    def step_failed(self, step_id: int, message: str = None) -> None:
        if step_id >= len(self.steps):
            return
        self.steps[step_id].set_end_timestamp()
        self.steps[step_id].set_failed(message)

    def method_call_failed(self, step_and_method_call_id: typing.Tuple[int, int], message: str = None) -> None:
        step_id, method_call_id = step_and_method_call_id
        if step_id >= len(self.steps):
            return
        primitive_step = self._get_primitive_step(step_id)
        primitive_step.set_method_call_end_timestamp(method_call_id)
        primitive_step.set_method_call_failed(method_call_id, message)

    def is_failed(self) -> bool:
        return self.status['state'] == metadata_base.PipelineRunStatusState.FAILURE.name

    def _set_problem(self, problem_description: problem.Problem) -> None:
        self.problem = {
            'id': problem_description['id'],
            'digest': problem_description.get_digest(),
        }

    def set_fold_group(self, fold_group_id: uuid.UUID, fold: int) -> None:
        self.run['fold_group'] = {
            'id': str(fold_group_id),
            'fold': fold,
        }

    def set_data_preparation_pipeline_run(
        self, data_preparation_pipeline_run: 'PipelineRun'
    ) -> None:
        if data_preparation_pipeline_run.start is None:
            raise exceptions.InvalidArgumentValueError("Data preparation pipeline start timestamp argument not provided.")

        if data_preparation_pipeline_run.end is None:
            raise exceptions.InvalidArgumentValueError("Data preparation pipeline end timestamp argument not provided.")

        self.run['data_preparation'] = {
            'pipeline': data_preparation_pipeline_run.pipeline,
            'steps': [step.to_json_structure() for step in data_preparation_pipeline_run.steps],
            'status': data_preparation_pipeline_run.status,
            'start': data_preparation_pipeline_run.start,
            'end': data_preparation_pipeline_run.end,
            'random_seed': data_preparation_pipeline_run.random_seed,
        }

        if data_preparation_pipeline_run.is_failed():
            message = 'Data preparation pipeline failed:\n{}'.format(
                data_preparation_pipeline_run.status['message']
            )
            self.status['state'] = metadata_base.PipelineRunStatusState.FAILURE.name
            if message is not None and message:
                self.status['message'] = message

    def set_scoring_pipeline_run(
        self, scoring_pipeline_run: 'PipelineRun', scoring_datasets: typing.Sequence[typing.Any] = None,
    ) -> None:
        if scoring_pipeline_run.start is None:
            raise exceptions.InvalidArgumentValueError("Scoring pipeline start timestamp argument not provided.")

        if scoring_pipeline_run.end is None:
            raise exceptions.InvalidArgumentValueError("Scoring pipeline end timestamp argument not provided.")

        self.run['scoring'] = {
            'pipeline': scoring_pipeline_run.pipeline,
            'steps': [step.to_json_structure() for step in scoring_pipeline_run.steps],
            'status': scoring_pipeline_run.status,
            'start': scoring_pipeline_run.start,
            'end': scoring_pipeline_run.end,
            'random_seed': scoring_pipeline_run.random_seed,
        }

        if scoring_datasets:
            self.run['scoring']['datasets'] = []
            for dataset in scoring_datasets:
                metadata = dataset.metadata.query(())
                self.run['scoring']['datasets'].append({
                    'id': metadata['id'],
                    'digest': metadata['digest'],
                })

        if scoring_pipeline_run.is_failed():
            message = 'Scoring pipeline failed:\n{}'.format(
                scoring_pipeline_run.status['message']
            )
            self.status['state'] = metadata_base.PipelineRunStatusState.FAILURE.name
            if message is not None and message:
                self.status['message'] = message

    def set_scores(
        self, scores: container.DataFrame, metrics: typing.Sequence[typing.Dict],
    ) -> None:
        if not self.is_standard_pipeline:
            raise exceptions.InvalidStateError("Setting scores for non-standard pipelines is not allowed.")

        json_scores = []

        if 'normalized' in scores.columns:
            columns = ['metric', 'value', 'normalized']
        else:
            columns = ['metric', 'value']

        for row in scores.loc[:, columns].itertuples(index=False, name=None):
            metric, value = row[:2]

            json_scores.append(
                {
                    # TODO: Why is "deepcopy" needed here?
                    'metric': copy.deepcopy(self._get_metric_description(metric, metrics)),
                    'value': float(value),
                },
            )

            if len(row) == 3:
                json_scores[-1]['normalized'] = float(row[2])

        if not json_scores:
            return

        if 'results' not in self.run:
            self.run['results'] = {}

        if 'scores' not in self.run['results']:
            self.run['results']['scores'] = json_scores
        else:
            raise exceptions.InvalidStateError("Scores already set for pipeline run.")

    def _get_metric_description(self, metric: str, performance_metrics: typing.Sequence[typing.Dict]) -> typing.Dict:
        """
        Returns a metric description from a list of them, given metric.

        Parameters
        ----------
        metric:
            A metric name.
        performance_metrics:
            A list of performance metric descriptions requested for scoring.

        Returns
        -------
        A metric description.
        """

        for performance_metric in performance_metrics:
            if performance_metric['metric'] == metric:
                metric_description = {
                    'metric': performance_metric['metric'].name,
                }

                if performance_metric.get('params', {}):
                    metric_description['params'] = performance_metric['params']

                return metric_description

        return {
            'metric': metric,
        }

    def set_predictions(self, predictions: container.DataFrame) -> None:
        if not self.is_standard_pipeline:
            raise exceptions.InvalidStateError("Setting predictions for non-standard pipelines is not allowed.")

        if not isinstance(predictions, container.DataFrame):
            logger.warning("Unable to set predictions for pipeline run because predictions are not a DataFrame.")
            return

        try:
            json_predictions: typing.Dict[str, typing.List] = {
                'header': [],
                'values': [],
            }

            column_names = []
            for column_index in range(len(predictions.columns)):
                # We use column name from the DataFrame is metadata does not have it. This allows a bit more compatibility.
                column_names.append(predictions.metadata.query_column(column_index).get('name', predictions.columns[column_index]))

                # "tolist" converts values to Python values and does not keep them as numpy.float64 or other special types.
                json_predictions['values'].append(utils.to_json_structure(predictions.iloc[:, column_index].tolist()))

            json_predictions['header'] += column_names

        except Exception as error:
            logger.warning("Unable to convert predictions to JSON structure for pipeline run.", exc_info=error)
            return

        if 'results' not in self.run:
            self.run['results'] = {}

        if 'predictions' not in self.run['results']:
            self.run['results']['predictions'] = json_predictions
        else:
            raise exceptions.InvalidStateError("Predictions already set for pipeline run.")

    def get_id(self) -> str:
        return self._to_json_structure()['id']

    @classmethod
    def json_structure_equals(cls, pipeline_run1: typing.Dict, pipeline_run2: typing.Dict) -> bool:
        """
        Checks whether two pipeline runs in a JSON structure are equal.
        This ignores the pipeline run id and all timestamps.
        """

        if not isinstance(pipeline_run1, collections.Mapping) or not isinstance(pipeline_run2, collections.Mapping):
            raise exceptions.InvalidArgumentTypeError("Pipeline run arguments must be dicts.")

        return utils.json_structure_equals(pipeline_run1, pipeline_run2, {'id', 'start', 'end', 'environment', 'logging'})


class RuntimeEnvironment(dict):
    def __init__(
        self, *,
        worker_id: str = None,
        cpu_resources: typing.Dict[str, typing.Any] = None,
        memory_resources: typing.Dict[str, typing.Any] = None,
        gpu_resources: typing.Dict[str, typing.Any] = None,
        reference_benchmarks: typing.Sequence[str] = None,
        reference_engine_version: str = None,
        engine_version: str = None,
        base_docker_image: typing.Dict[str, str] = None,
        docker_image: typing.Dict[str, str] = None,
    ) -> None:
        """
        Create an instance of the runtime environment description in which a pipeline is run.

        All values stored in an instance should be JSON compatible.

        Parameters
        ----------
        worker_id:
            A globally unique identifier for the machine on which the runtime is running.
            The idea is that multiple runs on the same system can be grouped together.
            If not provided, `uuid.getnode()` is used to obtain an identifier.
        cpu_resources:
            A description of the CPU resources available in this environment.
        memory_resources:
            A description of the memory resources available in this environment.
        gpu_resources:
            A description of the GPU resources available in this environment.
        reference_benchmarks:
            A list of ids of standard and optional additional benchmarks which were run in the same or
            equivalent RuntimeEnvironment. The timing characteristics of these benchmarks can be
            expected to be the same as anything timed in this RuntimeEnvironment.
        reference_engine_version:
            A git commit hash or version number for the reference engine used. If subclassing the
            reference engine, list it here.
        engine_version:
            A git commit hash or version number for the engine used. This is primarily useful for the
            author. If using the reference engine directly, list its git commit hash or version number
            here as well as in the reference_engine_version.
        base_docker_image:
            If the engine was run in a public or known docker container, specify the base docker image
            description here.
        docker_image:
            If the engine was run in a public or known docker container, specify the actual docker
            image description here. This is primarily useful for the author.
        """

        super().__init__()

        if worker_id is None:
            worker_id = self._get_worker_id()
        self['worker_id'] = worker_id

        resources = {}
        if cpu_resources is None:
            cpu_resources = self._get_cpu_resources()
        if cpu_resources is not None:
            resources['cpu'] = cpu_resources
        if memory_resources is None:
            memory_resources = self._get_memory_resources()
        if memory_resources is not None:
            resources['memory'] = memory_resources
        if gpu_resources is None:
            gpu_resources = self._get_gpu_resources()
        if gpu_resources is not None:
            resources['gpu'] = gpu_resources

        if resources:
            self['resources'] = resources

        if reference_benchmarks is not None:
            self['reference_benchmarks'] = reference_benchmarks

        if reference_engine_version is None:
            reference_engine_version = self._get_reference_engine_version()
        self['reference_engine_version'] = reference_engine_version

        if engine_version is None:
            engine_version = self['reference_engine_version']
        self['engine_version'] = engine_version

        if base_docker_image is None:
            base_docker_image = self._get_docker_image(
                environment_variables.D3M_BASE_IMAGE_NAME,
                environment_variables.D3M_BASE_IMAGE_DIGEST,
            )
        if base_docker_image is not None:
            self['base_docker_image'] = base_docker_image

        if docker_image is None:
            docker_image = self._get_docker_image(
                environment_variables.D3M_IMAGE_NAME,
                environment_variables.D3M_IMAGE_DIGEST,
            )
        if docker_image is not None:
            self['docker_image'] = docker_image

        # Here we assume that all values stored in "self" are JSON compatible.
        self['id'] = utils.compute_hash_id(self)

    @classmethod
    def _get_reference_engine_version(cls) -> str:
        try:
            # Get the git commit hash of the d3m repository.
            path = os.path.abspath(d3m.__file__).rsplit('d3m', 1)[0]
            return utils.current_git_commit(
                path=path, search_parent_directories=False,
            )
        except git.exc.InvalidGitRepositoryError:
            return d3m.__version__

    @classmethod
    def _get_worker_id(cls) -> str:
        """
        Compute the worker id.
        """

        mac_address = uuid.getnode()

        if mac_address >> 16 == DOCKER_MAC_ADDRESS_MASK >> 16:
            # Docker generates MAC addresses in the range 02:42:ac:11:00:00 to 02:42:ac:11:ff:ff
            # if one is not provided in the configuration
            logger.warning(
                "'worker_id' was generated using the MAC address inside Docker "
                "container and is not a reliable compute resource identifier."
            )
        elif (mac_address >> 40) % 2 == 1:
            # uuid.getnode docs state:
            # If all attempts to obtain the hardware address fail, we choose a
            # random 48-bit number with its eighth bit set to 1 as recommended
            # in RFC 4122.
            logger.warning(
                "'worker_id' was generated using a random number because the "
                "MAC address could not be determined."
            )

        return str(uuid.uuid5(WORKER_ID_NAMESPACE, json.dumps(mac_address, sort_keys=True)))

    @classmethod
    def _get_docker_image(cls, image_name_env_var: str, image_digest_env_var: str) -> typing.Optional[typing.Dict]:
        """
        Returns the docker image description.
        """

        docker_image = {}

        if image_name_env_var not in os.environ:
            logger.warning('Docker image environment variable not set: %(variable_name)s', {
                'variable_name': image_name_env_var,
            })
        elif os.environ[image_name_env_var]:
            docker_image['image_name'] = os.environ[image_name_env_var]

        if image_digest_env_var not in os.environ:
            logger.warning('Docker image environment variable not set: %(variable_name)s', {
                'variable_name': image_digest_env_var,
            })
        elif os.environ[image_digest_env_var]:
            docker_image['image_digest'] = os.environ[image_digest_env_var]

        if docker_image:
            return docker_image
        else:
            return None

    @classmethod
    def _get_configured(cls, environment_variable: str) -> typing.Optional[str]:
        if environment_variable not in os.environ:
            logger.warning('Configuration environment variable not set: %(variable_name)s', {
                'variable_name': environment_variable,
            })
            return None
        elif os.environ[environment_variable]:
            return os.environ[environment_variable]
        else:
            return None

    # TODO: Split into more methods.
    @classmethod
    def _get_cpu_resources(cls) -> typing.Optional[typing.Dict[str, typing.Any]]:
        cpu_resource: typing.Dict[str, typing.Any] = {}

        cpu_info: typing.Sequence[typing.Dict[str, str]] = []
        try:
            cpu_info = cls._read_info_file(PROC_CPU_PATH)
        except Exception as error:
            logger.warning(
                "Failed to get CPU information from '%(proc_cpu_path)s': %(error)s",
                {
                    'proc_cpu_path': PROC_CPU_PATH,
                    'error': error,
                },
            )

        # devices
        if cpu_info:
            cpu_resource['devices'] = [
                {
                    'name': cpu[PROC_CPU_MODEL_NAME_KEY],
                }
                for cpu in cpu_info
            ]

        # physical_present
        if cpu_info:
            physical_ids: typing.Set[str] = set()
            physical_present = 0
            for cpu in cpu_info:
                physical_id = cpu[PROC_CPU_PHYSICAL_ID_KEY]
                if physical_id in physical_ids:
                    continue
                physical_ids.add(physical_id)
                physical_present += int(cpu[PROC_CPU_CORES_KEY])
            cpu_resource['physical_present'] = physical_present

        # logical_present
        if cpu_info:
            cpu_resource['logical_present'] = len(cpu_info)

        # configured_available
        configured_available = cls._get_configured(
            environment_variables.D3M_CPU,
        )
        if configured_available is not None:
            cpu_resource['configured_available'] = configured_available

        # constraints
        constraints = {}
        try:
            with open(CGROUP_CPU_SHARES_PATH, 'r', encoding='ascii') as file:
                cpu_shares = int(file.read().strip())
                if cpu_shares < 1e5:
                    constraints['cpu_shares'] = cpu_shares
        except Exception as error:
            logger.warning(
                "Failed to get CPU information from '%(cgroup_cpu_shares_path)s': %(error)s",
                {
                    'cgroup_cpu_shares_path': CGROUP_CPU_SHARES_PATH,
                    'error': error,
                },
            )
        try:
            with open(CGROUP_CPU_CFS_PERIOD_US_PATH, 'r', encoding='ascii') as file:
                cfs_period_us = int(file.read().strip())
                constraints['cfs_period_us'] = cfs_period_us
        except Exception as error:
            logger.warning(
                "Failed to get CPU information from '%(cgroup_cpu_cfs_period_us_path)s': %(error)s",
                {
                    'cgroup_cpu_cfs_period_us_path': CGROUP_CPU_CFS_PERIOD_US_PATH,
                    'error': error,
                },
            )
        try:
            with open(CGROUP_CPU_CFS_QUOTA_US_PATH, 'r', encoding='ascii') as file:
                cfs_quota_us = int(file.read().strip())
                if cfs_quota_us >= 0:
                    constraints['cfs_quota_us'] = cfs_quota_us
        except Exception as error:
            logger.warning(
                "Failed to get CPU information from '%(cgroup_cpu_cfs_quota_us_path)s': %(error)s",
                {
                    'cgroup_cpu_cfs_quota_us_path': CGROUP_CPU_CFS_QUOTA_US_PATH,
                    'error': error,
                },
            )

        if 'cfs_period_us' in constraints and 'cfs_quota_us' not in constraints:
            del constraints['cfs_period_us']

        if constraints:
            cpu_resource['constraints'] = constraints

        if cpu_resource:
            return cpu_resource
        else:
            return None

    @classmethod
    def _read_info_file(cls, path: str) -> typing.Sequence[typing.Dict[str, str]]:
        info: typing.List[typing.Dict[str, str]] = [{}]

        with open(path, 'r', encoding='ascii') as file:
            for line in file:
                line = line.strip()
                if not line:
                    info.append({})
                    continue

                match = PROC_INFO_RE.match(line)
                if match is None:
                    raise ValueError("Error parsing.")

                key, value = match.groups()
                info[-1][key] = value

        if not info[-1]:
            del info[-1]

        return info

    # TODO: Split into more methods.
    # TODO: Get memory devices. Consider lshw.
    @classmethod
    def _get_memory_resources(cls) -> typing.Optional[typing.Dict[str, typing.Any]]:
        memory_resource: typing.Dict[str, typing.Any] = {}

        # total_memory (bytes)
        try:
            memory_info = cls._read_info_file(PROC_MEMORY_PATH)[0]
            total_memory_kb = int(memory_info[PROC_TOTAL_MEMORY_KEY].split()[0])
            memory_resource['total_memory'] = total_memory_kb * 1024
        except Exception as error:
            logger.warning(
                "Failed to get memory information from '%(proc_memory_path)s': %(error)s",
                {
                    'proc_memory_path': PROC_MEMORY_PATH,
                    'error': error,
                },
            )

        # configured_memory
        configured_memory = cls._get_configured(
            environment_variables.D3M_RAM,
        )
        if configured_memory is not None:
            memory_resource['configured_memory'] = configured_memory

        # constraints
        constraints = {}
        try:
            with open(CGROUP_MEMORY_LIMIT_PATH, 'r', encoding='ascii') as file:
                memory_limit = int(file.read().strip())
                if memory_limit < (sys.maxsize // 4096) * 4096:
                    constraints['memory_limit'] = memory_limit
        except FileNotFoundError:
            pass
        except Exception as error:
            logger.warning(
                "Failed to get memory information from '%(cgroup_memory_limit_path)s': %(error)s",
                {
                    'cgroup_memory_limit_path': CGROUP_MEMORY_LIMIT_PATH,
                    'error': error,
                },
            )

        if constraints:
            memory_resource['constraints'] = constraints

        if memory_resource:
            return memory_resource
        else:
            return None

    # TODO: Split into more methods.
    # TODO: Get GPU constraints.
    # TODO: Get GPU memory limit configuration.
    @classmethod
    def _get_gpu_resources(cls) -> typing.Optional[typing.Dict[str, typing.Any]]:
        gpu_resource: typing.Dict[str, typing.Any] = {}

        gpus: typing.List[GPUtil.GPU] = []
        try:
            gpus = GPUtil.getGPUs()
        except Exception as error:
            logger.warning(
                "Failed to get GPU information: %(error)s",
                {
                    'error': error,
                },
            )

        # devices
        if gpus:
            gpu_resource['devices'] = [
                {
                    'name': gpu.name,
                    'memory': int(gpu.memoryTotal) * 2**20,
                }
                for gpu in gpus
            ]

        # total_memory (bytes)
        if gpus:
            total_memory_mib = sum(gpu.memoryTotal for gpu in gpus)
            gpu_resource['total_memory'] = int(total_memory_mib) * 2**20

        if gpu_resource:
            return gpu_resource
        else:
            return None

    @classmethod
    def _yaml_representer(cls, dumper: yaml.Dumper, data: typing.Any) -> typing.Any:
        return dumper.represent_dict(data)


utils.yaml_add_representer(RuntimeEnvironment, RuntimeEnvironment._yaml_representer)


def _validate_pipeline_run_random_seeds(pipeline_run: typing.Dict) -> None:
    if 'random_seed' not in pipeline_run:
        raise exceptions.InvalidPipelineRunError("Pipeline run is missing a random seed.")

    if 'run' in pipeline_run:
        if 'data_preparation' in pipeline_run['run'] and 'random_seed' not in pipeline_run['run']['data_preparation']:
            raise exceptions.InvalidPipelineRunError("Data preparation pipeline run is missing a random seed.")

        if 'scoring' in pipeline_run['run'] and 'random_seed' not in pipeline_run['run']['scoring']:
            raise exceptions.InvalidPipelineRunError("Scoring pipeline run is missing a random seed.")

    for step in pipeline_run.get('steps', []):
        if step['type'] == 'SUBPIPELINE':
            _validate_pipeline_run_random_seeds(step)


def _validate_pipeline_run_timestamps(pipeline_run: typing.Dict, parent_start: datetime.datetime = None, parent_end: datetime.datetime = None) -> None:
    if 'start' not in pipeline_run:
        raise exceptions.InvalidPipelineRunError("Pipeline run is missing a start timestamp.")
    if 'end' not in pipeline_run:
        raise exceptions.InvalidPipelineRunError("Pipeline run is missing an end timestamp.")

    start = dateparser.parse(pipeline_run['start'], settings={'TIMEZONE': 'UTC'})
    end = dateparser.parse(pipeline_run['end'], settings={'TIMEZONE': 'UTC'})

    if start >= end:
        raise exceptions.InvalidPipelineRunError("Pipeline run contains a start timestamp which occurs after the corresponding end timestamp.")

    if parent_start is not None and parent_end is not None:
        if start <= parent_start or parent_end <= start:
            raise exceptions.InvalidPipelineRunError("Pipeline run contains a start timestamp which occurs outside the parent timestamp range.")

        if end <= parent_start or parent_end <= end:
            raise exceptions.InvalidPipelineRunError("Pipeline run contains an end timestamp which occurs outside the parent timestamp range.")

    for step in pipeline_run.get('steps', []):
        for method_call in pipeline_run.get('method_calls', []):
            _validate_pipeline_run_timestamps(method_call, start, end)

        _validate_pipeline_run_timestamps(step, start, end)

    if 'run' in pipeline_run:
        if 'data_preparation' in pipeline_run['run']:
            _validate_pipeline_run_timestamps(pipeline_run['run']['data_preparation'])

        if 'scoring' in pipeline_run['run']:
            _validate_pipeline_run_timestamps(pipeline_run['run']['scoring'])


def _validate_success_step(step: typing.Dict) -> None:
    if step['type'] == metadata_base.PipelineStepType.PRIMITIVE:
        for method_call in step.get('method_calls', []):
            if method_call['status']['state'] != metadata_base.PipelineRunStatusState.SUCCESS:
                raise exceptions.InvalidPipelineRunError(
                    "Step with '{expected_status}' status has a method call with '{status}' status".format(
                        expected_status=metadata_base.PipelineRunStatusState.SUCCESS,
                        status=method_call['status']['state'],
                    ),
                )
    elif step['type'] == metadata_base.PipelineStepType.SUBPIPELINE:
        _recurse_success(step)
    else:
        raise exceptions.UnexpectedValueError("Invalid pipeline run step type: {step_type}".format(step_type=step['type']))


def _validate_failure_step(step: typing.Dict) -> None:
    if step['type'] == metadata_base.PipelineStepType.PRIMITIVE:
        found_a_method_call_failure = False
        for method_call in step.get('method_calls', []):
            if found_a_method_call_failure:
                raise exceptions.InvalidPipelineRunError(
                    "There exists a method call after a method call with '{status}' status.".format(
                        status=metadata_base.PipelineRunStatusState.FAILURE,
                    ),
                )
            if method_call['status']['state'] == metadata_base.PipelineRunStatusState.FAILURE:
                found_a_method_call_failure = True
    elif step['type'] == metadata_base.PipelineStepType.SUBPIPELINE:
        _recurse_failure(step)
    else:
        raise exceptions.UnexpectedValueError("Invalid pipeline run step type: {step_type}".format(step_type=step['type']))


def _recurse_success(json_structure: typing.Dict) -> None:
    if 'steps' not in json_structure:
        raise exceptions.InvalidPipelineRunError("Successful pipeline run with missing steps.")

    for step in json_structure['steps']:
        if step['status']['state'] != metadata_base.PipelineRunStatusState.SUCCESS:
            raise exceptions.InvalidPipelineRunError(
                "Pipeline run with '{expected_status}' status has a step with '{status}' status".format(
                    expected_status=metadata_base.PipelineRunStatusState.SUCCESS,
                    status=step['status']['state'],
                ),
            )

        _validate_success_step(step)


def _recurse_failure(json_structure: typing.Dict) -> None:
    found_a_step_failure = False
    for step in json_structure.get('steps', []):
        if found_a_step_failure:
            raise exceptions.InvalidPipelineRunError(
                "There exists a step after a step with '{status}' status.".format(
                    status=metadata_base.PipelineRunStatusState.FAILURE,
                ),
            )

        if step['status']['state'] == metadata_base.PipelineRunStatusState.SUCCESS:
            _validate_success_step(step)
        elif step['status']['state'] == metadata_base.PipelineRunStatusState.FAILURE:
            found_a_step_failure = True
            _validate_failure_step(step)


def _validate_pipeline_run_status_consistency(pipeline_run: typing.Dict) -> None:
    """
    Verifies that the success or failure states of pipeline run components are consistent with each other.
    Any failure state should be propagated upwards to all parents in the pipeline run. The runtime should
    "short-circuit", meaning any failure state in the pipeline run should be the final component.
    """

    state = pipeline_run['status']['state']
    if state == metadata_base.PipelineRunStatusState.SUCCESS:
        _recurse_success(pipeline_run)
    elif state == metadata_base.PipelineRunStatusState.FAILURE:
        _recurse_failure(pipeline_run)
    else:
        raise exceptions.UnexpectedValueError("Invalid pipeline run state: {state}".format(state=state))


def _get_pipeline_run_references(pipeline_run: typing.Dict) -> typing.List[typing.Dict]:
    pipeline_run_references: typing.List[typing.Dict] = []

    pipeline_run_references += pipeline_run.get('environment', {}).get('reference_benchmarks', [])

    for step in pipeline_run.get('steps', []):
        pipeline_run_references += step.get('environment', {}).get('reference_benchmarks', [])

        for method_call in step.get('method_calls', []):
            pipeline_run_references += method_call.get('environment', {}).get('reference_benchmarks', [])

    return pipeline_run_references


def validate_pipeline_run(pipeline_run: typing.Dict) -> None:
    """
    Validates that the pipeline run is valid for the purpose of insertion in the metalearning database.
    If not, an exception is raised.

    Generally, metalearning database has additional requirements not captured by JSON schema.

    Parameters
    ----------
    pipeline_run:
        Pipeline run document.
    """

    PIPELINE_RUN_SCHEMA_VALIDATOR.validate(pipeline_run)

    if pipeline_run['schema'] != PIPELINE_RUN_SCHEMA_VERSION:
        raise exceptions.InvalidPipelineRunError(
            "Schema field is not '{expected_schema}', but '{actual_schema}'.".format(
                expected_schema=pipeline_module.PIPELINE_SCHEMA_VERSION,
                actual_schema=pipeline_run['schema'],
            ),
        )

    computed_id = utils.compute_hash_id(pipeline_run)

    if pipeline_run['id'] != computed_id:
        raise exceptions.InvalidPipelineRunError(
            "ID field is not '{computed_id}', but '{actual_id}'.".format(
                computed_id=computed_id,
                actual_id=pipeline_run['id'],
            ),
        )

    for dataset in list(pipeline_run['datasets']) + list(pipeline_run['run'].get('scoring', {}).get('datasets', [])):
        if set(dataset.keys()) != {'id', 'digest'}:
            raise exceptions.InvalidPipelineRunError("Invalid dataset reference: {dataset}".format(dataset=dataset))

    pipelines = [pipeline_run['pipeline']]
    if 'data_preparation' in pipeline_run['run']:
        pipelines.append(pipeline_run['run']['data_preparation']['pipeline'])
    if 'scoring' in pipeline_run['run']:
        pipelines.append(pipeline_run['run']['scoring']['pipeline'])

    for pipeline in pipelines:
        if set(pipeline.keys()) != {'id', 'digest'}:
            raise exceptions.InvalidPipelineRunError("Invalid pipeline reference: {pipeline}".format(pipeline=pipeline))

    if 'problem' in pipeline_run and set(pipeline_run['problem'].keys()) != {'id', 'digest'}:
        raise exceptions.InvalidPipelineRunError("Invalid problem reference: {problem}".format(problem=pipeline_run['problem']))

    referenced_pipeline_runs = []
    if 'previous_pipeline_run' in pipeline_run:
        referenced_pipeline_runs.append(pipeline_run['previous_pipeline_run'])
    referenced_pipeline_runs += _get_pipeline_run_references(pipeline_run)
    if 'scoring' in pipeline_run['run']:
        referenced_pipeline_runs += _get_pipeline_run_references(pipeline_run['run']['scoring'])
    if 'data_preparation' in pipeline_run['run']:
        referenced_pipeline_runs += _get_pipeline_run_references(pipeline_run['run']['data_preparation'])

    for referenced_pipeline_run in referenced_pipeline_runs:
        if set(referenced_pipeline_run.keys()) != {'id'}:
            raise exceptions.InvalidPipelineRunError("Invalid pipeline run reference: {pipeline_run}".format(pipeline_run=referenced_pipeline_run))

    _validate_pipeline_run_status_consistency(pipeline_run)
    _validate_pipeline_run_timestamps(pipeline_run)
    _validate_pipeline_run_random_seeds(pipeline_run)


def validate_pipeline(pipeline_description: typing.Dict) -> None:
    """
    Validates that the pipeline is valid for the purpose of insertion in the metalearning database.
    If not, an exception is raised.

    Generally, metalearning database has additional requirements not captured by JSON schema.

    Parameters
    ----------
    pipeline_description:
        Pipeline..
    """

    # Also validates against the schema. It validates top-level "digest" field if it exists.
    pipeline = pipeline_module.Pipeline.from_json_structure(pipeline_description, resolver=pipeline_module.NoResolver(strict_digest=True), strict_digest=True)

    if pipeline_description['schema'] != pipeline_module.PIPELINE_SCHEMA_VERSION:
        raise exceptions.InvalidPipelineError(
            "Schema field is not '{expected_schema}', but '{actual_schema}'.".format(
                expected_schema=pipeline_module.PIPELINE_SCHEMA_VERSION,
                actual_schema=pipeline_description['schema'],
            ),
        )

    # If there is "digest" field we know that it has matched the pipeline.
    if 'digest' not in pipeline_description:
        raise exceptions.InvalidPipelineError("Digest field is required.")

    # Also validates that there are no nested sub-pipelines.
    if pipeline_description != pipeline._canonical_pipeline_description(pipeline_description):
        raise exceptions.InvalidPipelineError("Pipeline description is not in canonical structure.")

    # We allow non-standard pipelines but require that all inputs are "Dataset" objects.
    input_types = {'inputs.{i}'.format(i=i): container.Dataset for i in range(len(pipeline.inputs))}
    pipeline.check(allow_placeholders=False, standard_pipeline=False, input_types=input_types)

    for step in pipeline.steps:
        if isinstance(step, pipeline_module.SubpipelineStep):
            # We are using "NoResolver", so we have "pipeline_description" available.
            if 'digest' not in step.pipeline_description:
                raise exceptions.InvalidPipelineError("Digest field in steps is required.")
        elif isinstance(step, pipeline_module.PrimitiveStep):
            # We are using "NoResolver", so we have "primitive_description" available.
            if 'digest' not in step.primitive_description:
                # A special case to handle a v2019.6.7 version of the core package where compute scores primitive
                # did not have a digest because it was lacking "installation" section in metadata.
                # See: https://gitlab.com/datadrivendiscovery/d3m/merge_requests/280
                if step.primitive_description['id'] == '799802fb-2e11-4ab7-9c5e-dda09eb52a70' and step.primitive_description['version'] == '0.3.0':
                    continue
                raise exceptions.InvalidPipelineError("Digest field in steps is required.")
        else:
            raise exceptions.InvalidPipelineError("Unknown step type: {type}".format(type=type(step)))


def validate_problem(problem_description_json_structure: typing.Dict) -> None:
    """
    Validates that the problem description is valid for the purpose of insertion in the metalearning database.
    If not, an exception is raised.

    Generally, metalearning database has additional requirements not captured by JSON schema.

    Parameters
    ----------
    problem_description_json_structure:
        Problem description as JSON structure.
    """

    if 'digest' not in problem_description_json_structure:
        raise exceptions.InvalidProblemError("Digest field is required.")

    # Also validates against the schema and checks the digest.
    problem_description = problem.Problem.from_json_structure(problem_description_json_structure, strict_digest=True)

    if problem_description['schema'] != problem.PROBLEM_SCHEMA_VERSION:
        raise exceptions.InvalidProblemError(
            "Schema field is not '{expected_schema}', but '{actual_schema}'.".format(
                expected_schema=problem.PROBLEM_SCHEMA_VERSION,
                actual_schema=problem_description['schema'],
            ),
        )

    canonical_problem_description = problem_description._canonical_problem_description(problem_description)

    if problem_description != canonical_problem_description:
        raise exceptions.InvalidProblemError("Problem description is not in canonical structure.")

    if problem_description.get('source', {}).get('from', {}).get('type', None) == 'REDACTED':
        problem_reference = problem_description['source']['from'].get('problem', {})
        if set(problem_reference.keys()) != {'id', 'digest'}:
            raise exceptions.InvalidProblemError("Invalid problem description reference for \"source.from.problem\": {problem}".format(problem=problem_reference))


def validate_dataset(dataset_description: typing.Dict) -> None:
    """
    Validates that the dataset description is valid for the purpose of insertion in the metalearning database.
    If not, an exception is raised.

    Generally, metalearning database has additional requirements not captured by JSON schema.

    Parameters
    ----------
    dataset_description:
        Dataset description.
    """

    metadata_base.CONTAINER_SCHEMA_VALIDATOR.validate(dataset_description)

    if dataset_description['schema'] != metadata_base.CONTAINER_SCHEMA_VERSION:
        raise exceptions.InvalidDatasetError(
            "Schema field is not '{expected_schema}', but '{actual_schema}'.".format(
                expected_schema=metadata_base.CONTAINER_SCHEMA_VERSION,
                actual_schema=dataset_description['schema'],
            ),
        )

    if 'id' not in dataset_description:
        raise exceptions.InvalidDatasetError("ID field is required.")

    if 'digest' not in dataset_description:
        raise exceptions.InvalidDatasetError("Digest field is required.")

    # Also validates that there are no nested sub-pipelines.
    if dataset_description != container.Dataset._canonical_dataset_description(dataset_description):
        raise exceptions.InvalidDatasetError("Dataset description is not in canonical structure.")

    if dataset_description['structural_type'] != 'd3m.container.dataset.Dataset':
        raise exceptions.InvalidDatasetError("Structural type is not 'd3m.container.dataset.Dataset', but '{type}'.".format(type=dataset_description['structural_type']))

    if dataset_description.get('source', {}).get('from', {}).get('type', None) == 'REDACTED':
        dataset_reference = dataset_description['source']['from'].get('dataset', {})
        if set(dataset_reference.keys()) != {'id', 'digest'}:
            raise exceptions.InvalidDatasetError("Invalid dataset reference for \"source.from.dataset\": {dataset}".format(dataset=dataset_reference))


def validate_primitive(primitive_json_structure: typing.Dict) -> None:
    """
    Validates that the primitive description is valid for the purpose of insertion in the metalearning database.
    If not, an exception is raised.

    Generally, metalearning database has additional requirements not captured by JSON schema.

    Parameters
    ----------
    primitive_json_structure:
        Primitive description as JSON structure.
    """

    if 'digest' not in primitive_json_structure:
        raise exceptions.InvalidProblemError("Digest field is required.")

    metadata_base.PrimitiveMetadata._validate(primitive_json_structure)


def pipeline_run_handler(arguments: argparse.Namespace) -> None:
    has_errored = False

    for pipeline_run_path in arguments.pipeline_runs:
        if getattr(arguments, 'list', False):
            print(pipeline_run_path)

        try:
            with utils.open(pipeline_run_path, 'r', encoding='utf8') as pipeline_run_file:
                if pipeline_run_path.endswith('.yml') or pipeline_run_path.endswith('.yaml') or pipeline_run_path.endswith('.yml.gz') or pipeline_run_path.endswith('.yaml.gz'):
                    pipeline_runs: typing.Iterator[typing.Dict] = utils.yaml_load_all(pipeline_run_file)
                else:
                    pipeline_runs = typing.cast(typing.Iterator[typing.Dict], [json.load(pipeline_run_file)])

                # It has to be inside context manager because YAML loader returns a lazy iterator
                # which requires an open file while iterating.
                for pipeline_run in pipeline_runs:
                    validate_pipeline_run(pipeline_run)
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=sys.stdout)
                print(f"Error validating a pipeline run: {pipeline_run_path}")
                has_errored = True
                continue
            else:
                raise Exception(f"Error validating a pipeline run: {pipeline_run_path}") from error

    if has_errored:
        sys.exit(1)


def pipeline_handler(
    arguments: argparse.Namespace, *, resolver_class: typing.Type[pipeline_module.Resolver] = None,
    no_resolver_class: typing.Type[pipeline_module.Resolver] = None, pipeline_class: typing.Type[pipeline_module.Pipeline] = None,
) -> None:
    has_errored = False

    for pipeline_path in arguments.pipelines:
        if getattr(arguments, 'list', False):
            print(pipeline_path)

        try:
            with utils.open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
                if pipeline_path.endswith('.yml') or pipeline_path.endswith('.yaml') or pipeline_path.endswith('.yml.gz') or pipeline_path.endswith('.yaml.gz'):
                    pipelines: typing.Iterator[typing.Dict] = utils.yaml_load_all(pipeline_file)
                else:
                    pipelines = typing.cast(typing.Iterator[typing.Dict], [json.load(pipeline_file)])

                for pipeline in pipelines:
                    validate_pipeline(pipeline)
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=sys.stdout)
                print(f"Error validating a pipeline: {pipeline_path}")
                has_errored = True
                continue
            else:
                raise Exception(f"Error validating a pipeline: {pipeline_path}") from error

    if has_errored:
        sys.exit(1)


def problem_handler(arguments: argparse.Namespace, *, problem_resolver: typing.Callable = None) -> None:
    has_errored = False

    for problem_path in arguments.problems:
        if getattr(arguments, 'list', False):
            print(problem_path)

        try:
            with utils.open(problem_path, 'r', encoding='utf8') as problem_file:
                if problem_path.endswith('.yml') or problem_path.endswith('.yaml') or problem_path.endswith('.yml.gz') or problem_path.endswith('.yaml.gz'):
                    problems: typing.Iterator[typing.Dict] = utils.yaml_load_all(problem_file)
                else:
                    problems = typing.cast(typing.Iterator[typing.Dict], [json.load(problem_file)])

                for problem in problems:
                    validate_problem(problem)
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=sys.stdout)
                print(f"Error validating a problem: {problem_path}")
                has_errored = True
                continue
            else:
                raise Exception(f"Error validating a problem: {problem_path}") from error

    if has_errored:
        sys.exit(1)


def dataset_handler(arguments: argparse.Namespace, *, dataset_resolver: typing.Callable = None) -> None:
    has_errored = False

    for dataset_path in arguments.datasets:
        if getattr(arguments, 'list', False):
            print(dataset_path)

        try:
            with utils.open(dataset_path, 'r', encoding='utf8') as dataset_file:
                if dataset_path.endswith('.yml') or dataset_path.endswith('.yaml'):
                    datasets: typing.Iterator[typing.Dict] = utils.yaml_load_all(dataset_file)
                else:
                    datasets = typing.cast(typing.Iterator[typing.Dict], [json.load(dataset_file)])

                for dataset in datasets:
                    validate_dataset(dataset)
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=sys.stdout)
                print(f"Error validating a dataset: {dataset_path}")
                has_errored = True
                continue
            else:
                raise Exception(f"Error validating a dataset: {dataset_path}") from error

    if has_errored:
        sys.exit(1)


def primitive_handler(arguments: argparse.Namespace) -> None:
    has_errored = False

    for primitive_path in arguments.primitives:
        if getattr(arguments, 'list', False):
            print(primitive_path)

        try:
            with utils.open(primitive_path, 'r', encoding='utf8') as primitive_file:
                if primitive_path.endswith('.yml') or primitive_path.endswith('.yaml'):
                    primitives: typing.Iterator[typing.Dict] = utils.yaml_load_all(primitive_file)
                else:
                    primitives = typing.cast(typing.Iterator[typing.Dict], [json.load(primitive_file)])

                for primitive in primitives:
                    validate_primitive(primitive)
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=sys.stdout)
                print(f"Error validating a primitive description: {primitive_path}")
                has_errored = True
                continue
            else:
                raise Exception(f"Error validating a primitive description: {primitive_path}") from error

    if has_errored:
        sys.exit(1)


if pyarrow_lib is not None:
    pyarrow_lib._default_serialization_context.register_type(
        PipelineRun, 'd3m.pipeline_run', pickle=True,
    )
