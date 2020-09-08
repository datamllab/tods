import abc
import typing

from d3m.metadata.problem import Problem, PerformanceMetric
from d3m.metadata.pipeline import Pipeline

from axolotl.utils.pipeline import PipelineResult
from axolotl.utils.schemas import ContainerType


class RunnerBase:
    """
    A base class for the pipeline runner backend.
    This child from this class must implement ``request_status`` and ``request_results`` which should keep
    track of all requests.

    Parameters
    ----------
    random_seed : int
        Random seed passed to the constructor.
    volumes_dir : str
        Path to a directory with static files required by primitives.
        In the standard directory structure (as obtained running ``python3 -m d3m index download``).
    scratch_dir : str
        Path to a directory to store any temporary files needed during execution.

    Attributes
    ----------
    random_seed : int
        Random seed passed to the constructor.
    volumes_dir : str
        Path to a directory with static files required by primitives.
        In the standard directory structure (as obtained running ``python3 -m d3m index download``).
    scratch_dir : str
        Path to a directory to store any temporary files needed during execution.
    """
    def __init__(self, *, random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None) -> None:
        self.random_seed = random_seed
        self.volumes_dir = volumes_dir
        self.scratch_dir = scratch_dir

    def add_metric(self, name: str, *, best_value: float, worst_value: float, score_class: type,
                   requires_confidence: bool = False, requires_rank: bool = False):
        """
        Method to register a new metric.

        Parameters
        ----------
        name : str
            Metric name, e.g. ACCURACY.
        best_value : float
            Value that represents the best e.g. in accuracy 1.0
        worst_value: float
            Value that represent the worst e.g. in accuracy 0
        score_class : type
            A class that helps computing the score.
        requires_confidence : bool
            A flag that tells if the scoring function requires a confidence value.
        requires_rank : bool
            A flag that tell if the scoring function requires the rank of the predictions.
        """

        PerformanceMetric.register_metric(name=name, best_value=best_value, worst_value=worst_value, score_class=score_class,
                                          requires_confidence=requires_confidence, requires_rank=requires_rank)

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    def fit_pipeline(self, problem_description: Problem, pipeline: Pipeline, input_data: typing.Sequence[ContainerType],
                     *, timeout: float = None, expose_outputs: bool = False) -> PipelineResult:
        """
        A method that fit a pipeline, save the state and returns a PipelineResult.

        Parameters
        ----------
        problem_description : Problem
            A problem description.
        pipeline : Pipeline
            A pipeline that are going to be fitted.
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.
        expose_outputs : bool
            A variable that enable exposing every intermediate results based on the input_data

        Returns
        -------
        PipelineResult
            A pipeline result containg the result of fitting the pipeline.
        """
        request_id = self.fit_pipeline_request(problem_description=problem_description, pipeline=pipeline,
                                               input_data=input_data, timeout=timeout,
                                               expose_outputs=expose_outputs)
        return self.get_request(request_id)

    @abc.abstractmethod
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

    # @abc.abstractmethod
    def produce_pipeline(self, fitted_pipeline_id: str, input_data: typing.Sequence[ContainerType], *,
                         timeout: float = None, expose_outputs: bool = False) -> PipelineResult:
        """
        A method that produce multiple fitted pipelines, save their state and returns a list of PipelineResult
        that contain the information of every pipeline run.

        Parameters
        ----------
        fitted_pipeline_id : str
            A list of fitted pipelines to run with the input_data
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.
        expose_outputs : bool
            A variable that enable exposing every intermediate results based on the input_data

        Returns
        -------
        PipelineResult
            A PipelineResult intance containing the information about the produced pipeline.
        """
        request_id = self.produce_pipeline_request(fitted_pipeline_id, input_data, timeout=timeout,
                                                   expose_outputs=expose_outputs)
        return self.get_request(request_id)

    @abc.abstractmethod
    def evaluate_pipeline_request(
            self, problem_description: Problem, pipeline: Pipeline,
            input_data: typing.Sequence[ContainerType], *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> str:
        """
        A method that evaluate multiple pipelines, and provides returns the scores and information of the pipelines.

        Parameters
        ----------
        problem_description : Problem
            A problem description.
        pipeline : Pipeline
            A list of pipelines that are going to be run.
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        metrics : typing.Sequence[typing.Dict]
            A dictionary containing the metrics and their arguments.
        data_preparation_pipeline : Pipeline
            A pipeline that prepares the data for the pipelines to be evaluated in, e.g. Cross-fold validation
        scoring_pipeline : Pipeline
            A pipeline that is used to compute the scores of the pipelines.
        data_preparation_params : typing.Dict[str, str]
            Parameters for the data preparation pipeline
        scoring_params: typing.Dict[str, str]
            Parameters for the scoring pipeline
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.

        Returns
        -------
        str
            A request id
        """

    def evaluate_pipeline(
            self, problem_description: Problem, pipeline: Pipeline,
            input_data: typing.Sequence[ContainerType], *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> PipelineResult:
        """
        A method that evaluate multiple pipelines, and provides returns the scores and information of the pipelines.

        Parameters
        ----------
        problem_description : Problem
            A problem description.
        pipeline : Pipeline
            A pipeline that is going to be evaluated.
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        metrics : typing.Sequence[typing.Dict]
            A dictionary containing the metrics and their arguments.
        data_preparation_pipeline : Pipeline
            A pipeline that prepares the data for the pipelines to be evaluated in, e.g. Cross-fold validation
        scoring_pipeline : Pipeline
            A pipeline that is used to compute the scores of the pipelines.
        data_preparation_params : typing.Dict[str, str]
            Parameters for the data preparation pipeline
        scoring_params: typing.Dict[str, str]
            Parameters for the scoring pipeline
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.

        Returns
        -------
        PipelineResult
           Result of the evaluation of the pipeline.
        """
        request_id = self.evaluate_pipeline_request(
            problem_description, pipeline, input_data, metrics=metrics,
            data_preparation_pipeline=data_preparation_pipeline, scoring_pipeline=scoring_pipeline,
            data_preparation_params=data_preparation_params, scoring_params=scoring_params, timeout=timeout
        )
        return self.get_request(request_id)

    def evaluate_pipelines(
            self, problem_description: Problem, pipelines: typing.Sequence[Pipeline],
            input_data: typing.Sequence[ContainerType], *, metrics: typing.Sequence[typing.Dict],
            data_preparation_pipeline: Pipeline = None, scoring_pipeline: Pipeline = None,
            data_preparation_params: typing.Dict[str, str] = None, scoring_params: typing.Dict[str, str] = None,
            timeout: float = None
    ) -> typing.Sequence[PipelineResult]:
        """
        A method that evaluate multiple pipelines, and provides returns the scores and information of the pipelines.

        Parameters
        ----------
        problem_description : Problem
            A problem description.
        pipelines : typing.Sequence[str]
            A list of pipelines that are going to be run.
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers.
        metrics : typing.Sequence[typing.Dict]
            A dictionary containing the metrics and their arguments.
        data_preparation_pipeline : Pipeline
            A pipeline that prepares the data for the pipelines to be evaluated in, e.g. Cross-fold validation
        scoring_pipeline : Pipeline
            A pipeline that is used to compute the scores of the pipelines.
        data_preparation_params : typing.Dict[str, str]
            Parameters for the data preparation pipeline
        scoring_params: typing.Dict[str, str]
            Parameters for the scoring pipeline
        timeout : float
            A maximum amount of time that pipelines are going to be executed in seconds.

        Returns
        -------
        typing.Sequence[PipelineResult]
           A sequence of PipelineResults.
        """
        request_ids = []
        for pipeline in pipelines:
            request_ids.append(
                self.evaluate_pipeline_request(
                    problem_description, pipeline, input_data, metrics=metrics,
                    data_preparation_pipeline=data_preparation_pipeline, scoring_pipeline=scoring_pipeline,
                    data_preparation_params=data_preparation_params, scoring_params=scoring_params, timeout=timeout
                )
            )

        return [self.get_request(request_id) for request_id in request_ids]
