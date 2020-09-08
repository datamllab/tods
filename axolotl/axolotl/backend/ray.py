import ray
import typing
import uuid
import binascii
import hashlib
import time
from ray.util import ActorPool

from d3m import index as d3m_index
from d3m import utils as d3m_utils
from d3m import runtime as runtime_module
from d3m.metadata.problem import Problem
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.base import Context
from d3m.metadata import pipeline_run as pipeline_run_module
from d3m import container as container_module

from axolotl.backend.base import RunnerBase
from axolotl.utils.pipeline import PipelineResult, save_pipeline_run, save_exposed_values
from axolotl.utils.schemas import ContainerType
import multiprocessing


@ray.remote
class DataHandler:
    def __init__(self):
        self.data = {}

    def add_data(self, input_data):
        if isinstance(input_data, list):
            values = []
            for _data in input_data:
                if isinstance(_data, container_module.Dataset):
                    values.append(_data.metadata.query(())['id'])

                data_id = str(hashlib.sha256(str(values).encode('utf8')).hexdigest())
                if data_id not in self.data:
                    self.data[data_id] = input_data
                return data_id

    def get_data(self, data_id):
        if data_id in self.data:
            return self.data[data_id]


@ray.remote
class RayExecutor:
    def __init__(self, *, random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None, store_results=False,
                 blocklist=()) -> None:
        self.random_seed = random_seed
        self.volumes_dir = volumes_dir
        self.scratch_dir = scratch_dir
        self.fitted_pipelines = {}
        with d3m_utils.silence():
            d3m_index.load_all(blocklist=blocklist)
            self.runtime_environment = pipeline_run_module.RuntimeEnvironment()
        self.store_results = store_results

    def fit_pipeline(
            self, data_handler, problem_description: Problem, pipeline:  Pipeline,
            input_data_id: str, *, timeout: float = None, expose_outputs: bool = False
    ) -> PipelineResult:
        pipeline_result = PipelineResult(pipeline=pipeline)
        pipeline_result.status = "RUNNING"
        pipeline_result.method_called = "fit"

        request_id = data_handler.get_data.remote(input_data_id)
        input_data = ray.get(request_id)

        is_standard_pipeline = False
        if len(input_data) == 1 and len(pipeline.outputs) == 1:
            is_standard_pipeline = True

        with d3m_utils.silence():
            runtime, output, result = runtime_module.fit(
                pipeline=pipeline, inputs=input_data, problem_description=problem_description, context=Context.TESTING,
                hyperparams=None, random_seed=self.random_seed, volumes_dir=self.volumes_dir,
                scratch_dir=self.scratch_dir,
                runtime_environment=self.runtime_environment, is_standard_pipeline=is_standard_pipeline,
                expose_produced_outputs=expose_outputs
            )

        if result.has_error():
            pipeline_result.status = "ERRORED"
            pipeline_result.error = result.error
        else:
            pipeline_result.status = "COMPLETED"
            fitted_pipeline_id = str(uuid.uuid4())

            if self.store_results:
                pipeline_result.exposed_outputs = save_exposed_values(result.values, pipeline.id, self.scratch_dir)
                pipeline_result.output = save_exposed_values(output, pipeline.id, self.scratch_dir)
            else:
                pipeline_result.exposed_outputs = result.values
                pipeline_result.output = output

            pipeline_result.fitted_pipeline_id = fitted_pipeline_id
            self.fitted_pipelines[fitted_pipeline_id] = runtime

        if self.store_results:
            pipeline_result.pipeline_run = save_pipeline_run(result.pipeline_run, self.scratch_dir)

        return pipeline_result

    def produce_pipeline(
            self, data_handler, fitted_pipeline_id: str, input_data_id: str, *,
            timeout: float = None, expose_outputs: bool = False
    ) -> PipelineResult:

        pipeline_result = PipelineResult(fitted_pipeline_id=fitted_pipeline_id)
        pipeline_result.status = "RUNNING"
        pipeline_result.method_called = "produce"
        pipeline_result.fitted_pipeline_id = fitted_pipeline_id

        request_id = data_handler.get_data.remote(input_data_id)
        input_data = ray.get(request_id)

        with d3m_utils.silence():
            output, result = runtime_module.produce(
                fitted_pipeline=self.fitted_pipelines[fitted_pipeline_id], test_inputs=input_data,
                expose_produced_outputs=expose_outputs
            )

        if result.has_error():
            pipeline_result.status = "ERRORED"
            pipeline_result.error = result.error
        else:
            pipeline_result.status = "COMPLETED"
            if self.store_results:
                pipeline_result.exposed_outputs = save_exposed_values(result.values, fitted_pipeline_id, self.scratch_dir)
                pipeline_result.output = save_exposed_values(output, fitted_pipeline_id, self.scratch_dir)
            else:
                pipeline_result.exposed_outputs = result.values
                pipeline_result.output = output

        if self.store_results:
            pipeline_result.pipeline_run = save_pipeline_run(result.pipeline_run, self.scratch_dir)

        return pipeline_result

    def evaluate_pipeline(
            self, data_handler, problem_description: Problem, pipeline: Pipeline,
            input_data_id: str, *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> PipelineResult:

        with d3m_utils.silence():
            pipeline_result = PipelineResult(pipeline=pipeline)
        pipeline_result.status = "RUNNING"
        pipeline_result.method_called = "evaluate"

        request_id = data_handler.get_data.remote(input_data_id)
        input_data = ray.get(request_id)

        with d3m_utils.silence():
            scores, results = runtime_module.evaluate(
                pipeline=pipeline, inputs=input_data, data_pipeline=data_preparation_pipeline,
                scoring_pipeline=scoring_pipeline, problem_description=problem_description,
                data_params=data_preparation_params, metrics=metrics, context=Context.TESTING,
                scoring_params=scoring_params, hyperparams=None, random_seed=self.random_seed,
                data_random_seed=self.random_seed, scoring_random_seed=self.random_seed,
                volumes_dir=self.volumes_dir, scratch_dir=self.scratch_dir, runtime_environment=self.runtime_environment
            )

        if results.has_error():
            pipeline_result.status = "ERRORED"
            pipeline_result.error = [result.error for result in results]
        else:
            pipeline_result.status = "COMPLETED"
            pipeline_result.scores = runtime_module.combine_folds(scores)

        if self.store_results:
            pipeline_result.pipeline_run = save_pipeline_run(results.pipeline_runs, self.scratch_dir)
        return pipeline_result

    def fitted_pipeline_id_exists(self, fitted_pipeline_id):
        return fitted_pipeline_id in self.fitted_pipelines


class RayRunner(RunnerBase):
    def __init__(self, *, random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None,
                 store_results=False, n_workers=None, blocklist=()) -> None:
        if not ray.is_initialized():
            ray.init()

        super().__init__(random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir)
        self.data_handler = DataHandler.remote()
        self.ray_executor = RayExecutor.remote(random_seed=random_seed,
                                               volumes_dir=volumes_dir, scratch_dir=scratch_dir,
                                               store_results=store_results,blocklist=blocklist)

        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        self.actor_pool = ActorPool([
            RayExecutor.remote(random_seed=random_seed, volumes_dir=volumes_dir,
                               scratch_dir=scratch_dir, store_results=store_results,
                               blocklist=blocklist) for _ in range(n_workers)]
        )

        # Wait for primitives to be load on the workers
        # time.sleep(len(d3m_index.search()) * 0.15)

    def stop_ray(self):
        ray.shutdown()

    def get_request(self, request_id: str):
        return ray.get(ray.ObjectID(binascii.unhexlify(request_id)))

    def fit_pipeline_request(self, problem_description: Problem, pipeline: Pipeline,
                             input_data: typing.Sequence[ContainerType], *, timeout: float = None,
                             expose_outputs: bool = False) -> str:

        request_id = self.data_handler.add_data.remote(input_data)
        input_data_id = ray.get(request_id)
        request_id = self.ray_executor.fit_pipeline.remote(self.data_handler, problem_description, pipeline, input_data_id,
                                                           timeout=timeout, expose_outputs=expose_outputs)
        return request_id.hex()

    def produce_pipeline_request(self, fitted_pipeline_id: str, input_data: typing.Sequence[ContainerType], *,
                                 timeout: float = None, expose_outputs: bool = False) -> str:
        request_id = self.data_handler.add_data.remote(input_data)
        input_data_id = ray.get(request_id)
        request_id = self.ray_executor.produce_pipeline.remote(self.data_handler, fitted_pipeline_id, input_data_id, timeout=timeout,
                                                               expose_outputs=expose_outputs)
        return request_id.hex()

    def evaluate_pipeline_request(
            self, problem_description: Problem, pipeline: Pipeline,
            input_data: typing.Sequence[ContainerType], *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> str:
        request_id = self.data_handler.add_data.remote(input_data)
        input_data_id = ray.get(request_id)

        request_id = self.ray_executor.evaluate_pipeline.remote(
            self.data_handler, problem_description, pipeline, input_data_id, metrics=metrics,
            data_preparation_pipeline=data_preparation_pipeline, scoring_pipeline=scoring_pipeline,
            data_preparation_params=data_preparation_params, scoring_params=scoring_params, timeout=timeout
        )
        return request_id.hex()

    def fitted_pipeline_id_exists(self, fitted_pipeline_id):
        request_id = self.ray_executor.fitted_pipeline_id_exists.remote(fitted_pipeline_id)
        return ray.get(request_id)

    def evaluate_pipelines(
            self, problem_description: Problem, pipelines: typing.Sequence[Pipeline],
            input_data: typing.Sequence[ContainerType], *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> typing.Sequence[PipelineResult]:
        request_id = self.data_handler.add_data.remote(input_data)
        input_data_id = ray.get(request_id)

        args = []
        for pipeline in pipelines:
            args.append({
                'data_handler': self.data_handler, 'problem_description': problem_description, 'pipeline': pipeline,
                'input_data_id': input_data_id, 'metrics': metrics, 'data_preparation_pipeline': data_preparation_pipeline,
                'scoring_pipeline': scoring_pipeline,'data_preparation_params': data_preparation_params,
                'scoring_params': scoring_params,'timeout': timeout
            })

        return self.actor_pool.map(lambda actor, arg: actor.evaluate_pipeline.remote(**arg), args)
