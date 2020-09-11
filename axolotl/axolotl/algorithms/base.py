import abc
import uuid
import logging
import time
import typing

from d3m.metadata.problem import Problem
from d3m.metadata.pipeline import Pipeline
from d3m import runtime as runtime_module
from d3m import container
from d3m.metadata.base import Context
from d3m import utils as d3m_utils
from d3m.metadata import pipeline_run as pipeline_run_module

from axolotl.backend.base import RunnerBase
from axolotl.utils.pipeline import PipelineResult
from axolotl.utils.schemas import ContainerType
from axolotl.utils import resources as resources_module

logger = logging.getLogger(__name__)


class PipelineSearchBase:
    """
    Base class for pipeline searcher, this class should provide the common interface for pipeline
    searchers to be integrated with the system.

    Nothing should be computed or initialized on the constructor, just adding more variables.
    Everything else should be computed at start_search.

    Parameters
    ----------
    problem_description : Problem
        A problem description.
    backend : RunnerBase
        An instance of a backend class.
    primitives_blocklist : typing.Sequence[str]
        A list of string with pipeline names to avoid.
    ranking_function : typing.Callable
        A function that takes as an input a dataframe of scores, and generates a rank, smaller is better


    Attributes
    ----------
    backend : RunnerBase
        An instance of a backend class.
    random_seed : int
        Random seed passed to the constructor.
    volumes_dir : str
        Path to a directory with static files required by primitives.
    scratch_dir : str
        Path to a directory to store any temporary files needed during execution.
    ranking_function : typing.Callable
        A function that takes as an input a dataframe of scores, and generates a rank, smaller is better
    problem_description : Problem
        A problem description.
    primitives_blocklist : typing.Sequence[str]
        A list of string with pipeline names to avoid.

    history : typing.Dict[str, PipelineResult]
        A list of all the evaluated pipelines with their execution results and performance.
    """

    def __init__(self,
                 problem_description: Problem, backend: RunnerBase, *,
                 primitives_blocklist: typing.Sequence[str] = None, ranking_function: typing.Callable = None
        ) -> None:
        self.search_id = str(uuid.uuid4())
        self.backend = backend
        self.random_seed = backend.random_seed
        self.volumes_dir = backend.volumes_dir
        self.scratch_dir = backend.scratch_dir
        self.ranking_function = ranking_function

        self.problem_description: Problem = problem_description
        self.primitives_blocklist: typing.Sequence[str] = primitives_blocklist

        self.history: typing.List[PipelineResult] = []

        # missing typing
        self.best_fitted_pipeline_id: str = None
        self.input_data: typing.Sequence[ContainerType] = None

        with d3m_utils.silence():
            self.runtime_environment = pipeline_run_module.RuntimeEnvironment()

    def search(self, time_limit: float):
        """
        This method executes the whole search, by calling the ``_search`` method multiple times
        as long as there is time left and put the results on the history.

        Parameters
        ----------
        time_limit : float
            Time limit for the search
        """
        time_start = time.time()
        largest_iteration = 0

        i = 0

        while True:
            i += 1
            time_left = time_limit - (time.time() - time_start)

            if time_left < 5:
                logger.info('-- Time out --')
                break

            if time_left - largest_iteration < 5:
                logger.info("""-- Time out -- \n Time left {} Next iteration could be over {}""".format(time_left, largest_iteration))
                break

            start_iteration_time = time.time()
            results = self._search(time_left=time_left)
            self.history += results
            current_iteration_time = time.time() - start_iteration_time

            if largest_iteration < current_iteration_time:
                largest_iteration = current_iteration_time

    def search_fit(self, input_data: typing.Sequence[ContainerType], time_limit: float = 300, *,
                   expose_values: bool = False) -> typing.Tuple[runtime_module.Runtime, PipelineResult]:
        """
        This method calls search and fit the best ranking pipelines located from the search located on the history.

        Parameters
        ----------
        input_data : typing.Sequence[ContainerType]
            A list of D3M containers to be use as the pipeline input.

        time_limit : float
            The time limit to be use for the search.

        expose_values : bool
            A flag that allows the user expose all intermediate result of the pipeline during fitting.
        """
        self.input_data = input_data
        self.search(time_limit)

        best_pipeline = None
        for pipeline_result in self.history:
            if pipeline_result.error is None:
                if best_pipeline is None:
                    best_pipeline = pipeline_result
                else:
                    if pipeline_result.rank < best_pipeline.rank:
                        best_pipeline = pipeline_result

        if best_pipeline is None:
            logging.error('No solution founded')
            pipeline_result = PipelineResult(fitted_pipeline_id='')
            pipeline_result.error = RuntimeError("No solution found")
            return None, pipeline_result

        return self.fit(best_pipeline.pipeline, input_data, expose_values)

    def fit(self, pipeline: Pipeline, input_data: typing.Sequence[container.Dataset],
            expose_outputs: bool = False) -> typing.Tuple[runtime_module.Runtime, PipelineResult]:

        pipeline_result = PipelineResult(pipeline=pipeline)

        runtime, output, result = runtime_module.fit(
            pipeline=pipeline, inputs=input_data, problem_description=self.problem_description, context=Context.TESTING,
            hyperparams=None, random_seed=self.random_seed, volumes_dir=self.volumes_dir,
            runtime_environment=self.runtime_environment, scratch_dir=self.scratch_dir, expose_produced_outputs=expose_outputs
        )
        if result.has_error():
            pipeline_result.status = "ERRORED"
            pipeline_result.error = result.error
        else:
            pipeline_result.status = "COMPLETED"

            pipeline_result.exposed_outputs = result.values
            pipeline_result.output = output

        return runtime, pipeline_result

    def produce(self, fitted_pipeline: runtime_module.Runtime, input_data: typing.Sequence[container.Dataset],
                expose_outputs: bool = False) -> PipelineResult:
        pipeline_result = PipelineResult(fitted_pipeline_id='')

        with d3m_utils.silence():
            output, result = runtime_module.produce(
                fitted_pipeline=fitted_pipeline, test_inputs=input_data,
                expose_produced_outputs=expose_outputs
            )

        if result.has_error():
            pipeline_result.status = "ERRORED"
            pipeline_result.error = result.error
        else:
            pipeline_result.status = "COMPLETED"

            pipeline_result.exposed_outputs = result.values
            pipeline_result.output = output
        return pipeline_result

    @abc.abstractmethod
    def _search(self, time_left: float) -> typing.Sequence[PipelineResult]:
        """
        A method where the search is going to be implemented.
        The search algorithm should be iteration oriented, each of the call should end
        on returning the status of pipelines evaluated.

        Parameters
        ----------
        time_left : float
            TTime left for the iteration

        Returns
        -------
         typing.Sequence[PipelineResult]
            A list of pipeline results with the information of the pipeline ran during the iteration.

        """

    def pretty_print(self, deep: bool = False):
        """
        A function that prints everything really nice.
        """
        from pprint import pprint

        def simplify_value(input_value):
            if isinstance(input_value, Problem):
                return input_value.to_simple_structure()
            elif isinstance(input_value, Pipeline):
                return input_value.to_json_structure()
            elif isinstance(input_value, PipelineResult):
                return vars(input_value)
            elif isinstance(input_value, dict):
                new_value = {}
                for nested_variable, nested_val in input_value.items():
                    new_value[nested_variable] = simplify_value(nested_val)
                return new_value

        class_instance = vars(self)
        if deep:
            class_instance = simplify_value(class_instance)

        pprint(class_instance)
