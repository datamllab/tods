import typing
import uuid

from d3m import utils as d3m_utils
from d3m import runtime as runtime_module
from d3m.metadata.problem import Problem
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.base import Context
from d3m.metadata import pipeline_run as pipeline_run_module

from axolotl.backend.base import RunnerBase
from axolotl.utils.pipeline import PipelineResult
from axolotl.utils.schemas import ContainerType


class SimpleRunner(RunnerBase):
    def __init__(self, *, random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None) -> None:
        super().__init__(random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir)
        self.fitted_pipelines = {}
        self.request_results = {}

        with d3m_utils.silence():
            self.runtime_environment = pipeline_run_module.RuntimeEnvironment()

    def get_request(self, request_id: str) -> PipelineResult:
        """
        A method that returns the result from the requests

        Parameters
        ----------
        request_id : str
            Request id of data to retrieve

        Returns
        -------
        PipelineResult
            A PipelineResult instance that contains the information.
        """
        if request_id in self.request_results:
            return self.request_results[request_id]
        else:
            return PipelineResult(fitted_pipeline_id='')

    def fit_pipeline_request(self, problem_description: Problem, pipeline: Pipeline,
                             input_data: typing.Sequence[ContainerType], *, timeout: float = None,
                             expose_outputs: bool = False) -> str:
        """
        A method that submit a fit_pipeline job.

        Parameters
        ----------
        problem_description : Problem
            A problem description.
        pipeline : Pipeline
            The pipeline that is going to be fitted.
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.
        expose_outputs : bool
            A variable that enable exposing every intermediate results based on the input_data

        Returns
        -------
        str
            A request id.
        """
        request_id = str(uuid.uuid4())
        pipeline_result = PipelineResult(pipeline=pipeline)
        pipeline_result.status = "RUNNING"
        pipeline_result.method_called = "fit"

        is_standard_pipeline = False
        if len(input_data) == 1 and len(pipeline.outputs) == 1:
            is_standard_pipeline = True

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
            pipeline_result.exposed_outputs = result.values
            pipeline_result.output = output
            fitted_pipeline_id = str(uuid.uuid4())
            pipeline_result.fitted_pipeline_id = fitted_pipeline_id
            self.fitted_pipelines[fitted_pipeline_id] = runtime

        pipeline_result.pipeline_run = result.pipeline_run
        self.request_results[request_id] = pipeline_result

        return request_id

    def produce_pipeline_request(self, fitted_pipeline_id: str, input_data: typing.Sequence[ContainerType], *,
                                 timeout: float = None, expose_outputs: bool = False) -> str:
        """
        A method that submit a produce pipeline request.

        Parameters
        ----------
        fitted_pipeline_id : str
            The fitted pipeline if of the fitted pipeline to be use to produce results.
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.
        expose_outputs : bool
            A variable that enable exposing every intermediate results based on the input_data

        Returns
        -------
        str
            A request id.
        """
        request_id = str(uuid.uuid4())

        pipeline_result = PipelineResult(fitted_pipeline_id=fitted_pipeline_id)
        pipeline_result.status = "RUNNING"
        pipeline_result.method_called = "produce"
        pipeline_result.fitted_pipeline_id = fitted_pipeline_id

        output, result = runtime_module.produce(
            fitted_pipeline=self.fitted_pipelines[fitted_pipeline_id], test_inputs=input_data,
            expose_produced_outputs=expose_outputs
        )

        if result.has_error():
            pipeline_result.status = "ERRORED"
            pipeline_result.error = result.error
        else:
            pipeline_result.status = "COMPLETED"
            pipeline_result.output = output
            pipeline_result.exposed_outputs = result.values

        pipeline_result.pipeline_run = result.pipeline_run
        self.request_results[request_id] = pipeline_result

        return request_id

    def evaluate_pipeline_request(
            self, problem_description: Problem, pipeline: Pipeline,
            input_data: typing.Sequence[ContainerType], *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> str:
        request_id = str(uuid.uuid4())

        pipeline_result = PipelineResult(pipeline=pipeline)
        pipeline_result.status = "RUNNING"
        pipeline_result.method_called = "evaluate"

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
            pipeline_result.outputs = [result.values for result in results]

        self.request_results[request_id] = pipeline_result
        return request_id

