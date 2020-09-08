import argparse
import json
import logging
import os
import pathlib
import time
import warnings
from concurrent import futures
import ray
import os
import uuid

import google.protobuf.timestamp_pb2 as p_timestamp
import grpc
from d3m import utils as d3m_utils, index as d3m_index
from d3m.metadata import problem as problem_module
from d3m.metadata.pipeline import Resolver
from d3m import container
from d3m import runtime as runtime_module
from d3m.metadata.base import Context
from ta3ta2_api import core_pb2, core_pb2_grpc, primitive_pb2, value_pb2, utils

from axolotl.backend.ray import RayRunner
from axolotl.algorithms.dummy import DummySearch, dummy_ranking_function
from axolotl.algorithms.data_driven_search import DataDrivenSearch
from axolotl.utils.pipeline import load_pipeline, save_pipeline
from axolotl.d3m_grpc.constants import SearchPath, EnvVars, PrimitivesList, Path
from axolotl.utils import resources as resources_module, schemas as schemas_utils

from pprint import pprint


__version__ = '2020.4.4_pre'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

logger = logging.getLogger(__name__)
AGENT = 'TAMU.10.0_pre'
ALLOWED_VALUE_TYPES = ['RAW', 'DATASET_URI', 'CSV_URI']
SUPPORTED_EXTENSIONS = []


def available_primitives():
    primitives_info = []

    with d3m_utils.silence():
        for primitive_path in d3m_index.search():
            if primitive_path in PrimitivesList.BlockList:
                continue

            try:
                primitive = d3m_index.get_primitive(primitive_path)
                primitive_id = primitive.metadata.query()['id']
                version = primitive.metadata.query()['version']
                python_path = primitive.metadata.query()['python_path']
                name = primitive.metadata.query()['name']
                digest = primitive.metadata.query().get('digest', None)
                primitive_info = {
                    'id': primitive_id,
                    'version': version,
                    'python_path': python_path,
                    'name': name,
                    'digest': digest
                }
                primitives_info.append(primitive_info)
            except:
                continue
    return primitives_info


PRIMITIVES_LIST = available_primitives()


@ray.remote
class SearchWrappers:
    def __init__(self, search_class, problem_description, backend, primitives_blocklist=None, ranking_function=None, n_workers=2):
        self.search_algorithm = search_class(problem_description=problem_description, backend=backend,
                                             primitives_blocklist=primitives_blocklist, ranking_function=ranking_function,
                                             n_workers=n_workers)
        self._seen_index = 0
        self.has_input_data = False
        self.time_left = None
        self.active_search = True
        self.save_path = SearchPath(self.search_algorithm.search_id)

    def search_request(self, time_left, input_data=None):
        time_start = time.time()
        if not self.has_input_data:
            self.search_algorithm.input_data = input_data
            self.time_left = time_left
            self.has_input_data = True

        results = self.search_algorithm._search(time_left)
        self.search_algorithm.history += results
        succeed_pipelines = []
        for result in results:
            print('pipeline', result.pipeline.id, result.status)
            # save all results in pipelines searched
            save_pipeline(result.pipeline, self.save_path.pipelines_searched)

            # save all pipelines_runs
            resources_module.copy_file(result.pipeline_run, self.save_path.pipeline_runs)

            # we filter the ones that were completed
            if result.status == 'COMPLETED':
                # since we were able to score it, we put a copy into the pipelines_scored directory
                save_pipeline(result.pipeline, self.save_path.pipelines_scored)
                succeed_pipelines.append(result)

        self.time_left -= time.time() - time_start
        return succeed_pipelines

    def end_search(self):
        self.active_search = False

    def is_search_active(self):
        return self.active_search

    def get_search_id(self):
        return self.search_algorithm.search_id

    def get_time_left(self):
        return self.time_left


class Core(core_pb2_grpc.CoreServicer):
    """
    A class that works as a server that provides support for the pipeline searches, and provides the interfaces
    defined on the TA3-2 API.

    Attributes
    ----------
    version: str
        A str that represents the version of the Ta3-2 api that is supporting.
    user_agents: dict()
        A simple dictionary that keep the relation of the different users.
    manager: ExecutionManger
        Schedules the searches, and all resources related with the search.
    """

    def __init__(self):
        logger.info('########## Initializing Service ##########')
        self.version = core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version]
        self.n_workers = EnvVars.D3MCPU
        if self.n_workers > 7:
            self.n_workers = int(self.n_workers/2) + 1
        print('Server n_workers', self.n_workers)
        self.backend = RayRunner(random_seed=0, volumes_dir=EnvVars.D3MSTATICDIR, scratch_dir=Path.TEMP_STORAGE_ROOT,
                                 blocklist=PrimitivesList.BlockList, store_results=True, n_workers=self.n_workers)
        self.searches = {}
        self.request_mapping = {}
        self.solutions = {}
        self.problem_descriptions = {}

    # TODO add support for templates
    def SearchSolutions(self, request, context):
        user_agent = request.user_agent
        logger.info('method=SearchSolution, agent=%s', user_agent)

        # Checking version of protocol.
        if request.version != self.version:
            logger.info(' method=SearchSolution, info=Different api version%s', self.version)

        # Types allowed by client
        allowed_value_types = list(request.allowed_value_types)

        if not allowed_value_types:
            allowed_value_types = ALLOWED_VALUE_TYPES

        problem_description = utils.decode_problem_description(request.problem)

        # Parsing and storing Pipeline Template (store this to a file instead of passing it)
        with d3m_utils.silence():
            template = utils.decode_pipeline_description(pipeline_description=request.template,
                                                         resolver=Resolver(primitives_blocklist=PrimitivesList.BlockList))

        time_bound_search = request.time_bound_search
        time_bound_search = time_bound_search * 60

        input_data = [load_data(utils.decode_value(x)) for x in request.inputs]

        search = SearchWrappers.remote(search_class=DataDrivenSearch, problem_description=problem_description,
                                       backend=self.backend, primitives_blocklist=PrimitivesList.BlockList,
                                       ranking_function=dummy_ranking_function, n_workers=self.n_workers)

        request_id = search.get_search_id.remote()
        search_id = ray.get(request_id)

        # print('got search_id')
        self.searches[search_id] = search
        request_id = self.searches[search_id].search_request.remote(time_left=time_bound_search, input_data=input_data)

        self.request_mapping[search_id] = request_id
        self.solutions[search_id] = []
        self.problem_descriptions[search_id] = problem_description
        response = core_pb2.SearchSolutionsResponse(search_id=search_id)
        return response

    def GetSearchSolutionsResults(self, request, context):
        search_id = request.search_id
        logger.info('method=GetSearchSolutionsResults, search_id=%s', search_id)
        request_id = self.request_mapping[search_id]

        progress_start = p_timestamp.Timestamp()
        progress_end = p_timestamp.Timestamp()

        all_ticks = 0
        done_ticks = 0

        # Yield running so the client know the search is running.
        progress = core_pb2.Progress(state='RUNNING', status='Running Search', start=progress_start)
        response = core_pb2.GetSearchSolutionsResultsResponse(progress=progress)
        yield response

        has_solution = False

        succeed_pipelines = ray.get(request_id)
        time_left_id = self.searches[search_id].get_time_left.remote()
        time_left = ray.get(time_left_id)

        while True:
            start_time = time.time()

            # if no time left we stop
            if time_left < 5:
                break

            # case if a signal from EndSolution is sent to stop the search
            is_active_id = self.searches[search_id].is_search_active.remote()
            is_active = ray.get(is_active_id)

            if not is_active:
                logger.info('method=GetSearchSolutionsResults, search_id={} message=SearchStopped'.format(search_id))
                break

            for succeed_pipeline in succeed_pipelines:
                has_solution = True
                logger.info('method=GetSearchSolutionsResults, search_id={} solution_id={}'.format(
                    search_id,succeed_pipeline.pipeline.id))
                response = core_pb2.GetSearchSolutionsResultsResponse(
                    progress=progress,
                    done_ticks=done_ticks,
                    all_ticks=all_ticks,
                    solution_id=succeed_pipeline.pipeline.id,
                    internal_score=1-succeed_pipeline.rank,
                    scores=[core_pb2.SolutionSearchScore(scores=encode_scores(succeed_pipeline))]
                )
                self.solutions[search_id].append(succeed_pipeline.pipeline.id)
                yield response

            finished, running = ray.wait([request_id], timeout=1)

            if finished:
                succeed_pipelines = ray.get(request_id)
                request_id = self.searches[search_id].search_request.remote(time_left=time_left)
            else:
                succeed_pipelines = []

            time.sleep(1)

            time_left -= time.time() - start_time

        if has_solution:
            progress_state = 'COMPLETED'
            progress_status = 'Search completed'
        else:
            progress_state = 'ERRORED'
            progress_status = 'No solution founded'

        logger.info('method=GetSearchSolutionsResults, search_id={}, status={}, message={}'.format(
            search_id, progress_state, progress_status)
        )
        progress_end.GetCurrentTime()
        progress = core_pb2.Progress(state=progress_state, status=progress_status,
                                     start=progress_start, end=progress_end)
        response = core_pb2.GetSearchSolutionsResultsResponse(progress=progress, done_ticks=done_ticks,
                                                              all_ticks=all_ticks,)
        yield response

    def EndSearchSolutions(self, request, context):
        search_id = request.search_id
        logger.info('method=EndSearchSolutions search_id=%s', search_id)
        ray.kill(self.searches[search_id])
        del self.searches[search_id]
        response = core_pb2.EndSearchSolutionsResponse()
        return response

    def StopSearchSolutions(self, request, context):
        search_id = request.search_id
        self.searches[search_id].end_search.remote()
        logger.info('method=StopSearchSolutions search_id=%s', search_id)
        response = core_pb2.StopSearchSolutionsResponse()
        return response

    def DescribeSolution(self, request, context):
        solution_id = request.solution_id
        logger.info('method=DescribeSolution, solution_id=%s', solution_id)

        pipeline, _, _ = self.get_solution_problem(solution_id)
        if pipeline is None:
            logger.info('method=DescribeSolution, solution_id=%s, error=Solution_id not found', solution_id)
            response = core_pb2.DescribeSolutionResponse()
            return response

        with d3m_utils.silence():
            pipeline = utils.encode_pipeline_description(pipeline, ALLOWED_VALUE_TYPES, Path.TEMP_STORAGE_ROOT)

        response = core_pb2.DescribeSolutionResponse(pipeline=pipeline)
        return response

    def ScoreSolution(self, request, context):
        solution_id = request.solution_id
        logger.info('method=SocreSolution, solution_id=%s', solution_id)

        pipeline, problem_description, _ = self.get_solution_problem(solution_id)
        if pipeline is None:
            logger.info('method=FitSolution, solution_id=%s, status=ERRORED, error=Solution_id not found', solution_id)
            response = core_pb2.ScoreSolutionResponse()
            return response

        input_data = [load_data(utils.decode_value(x)) for x in request.inputs]
        metrics = [utils.decode_performance_metric(metric) for metric in request.performance_metrics]
        scoring_pipeline = schemas_utils.get_scoring_pipeline()
        data_preparation_params = decode_scoring_configuration(request.configuration)
        data_preparation_pipeline = schemas_utils.get_splitting_pipeline(data_preparation_params['method'])

        request_id = self.backend.evaluate_pipeline_request(
            problem_description=problem_description, pipeline=pipeline, input_data=input_data,
            metrics=metrics, data_preparation_pipeline=data_preparation_pipeline,
            scoring_pipeline=scoring_pipeline, data_preparation_params=data_preparation_params)

        response = core_pb2.ScoreSolutionResponse(request_id=request_id)
        return response

    def GetScoreSolutionResults(self, request, context):
        request_id = request.request_id
        logger.info('method=GetScoreSolutionResults, request_id=%s', request_id)

        progress_start = p_timestamp.Timestamp()
        progress_end = p_timestamp.Timestamp()
        progress_start.GetCurrentTime()

        progress = core_pb2.Progress(state='RUNNING', status='Running score job', start=progress_start)
        response = core_pb2.GetScoreSolutionResultsResponse(progress=progress)
        yield response

        pipeline_result = self.backend.get_request(request_id)
        progress_end.GetCurrentTime()

        if pipeline_result.error is None:
            progress = core_pb2.Progress(
                state='COMPLETED',
                status='Score job COMPLETED',
                start=progress_start,
                end=progress_end
            )

            response = core_pb2.GetScoreSolutionResultsResponse(
                progress=progress, scores=encode_scores(pipeline_result))
        else:
            progress = core_pb2.Progress(
                state='ERRORED',
                status=str(pipeline_result.error),
                start=progress_start,
                end=progress_end
            )

            response = core_pb2.GetScoreSolutionResultsResponse(progress=progress)
        yield response
        return

    def FitSolution(self, request, context):
        solution_id = request.solution_id
        logger.info('method=FitSolution solution_id=%s', solution_id)

        pipeline, problem_description, _ = self.get_solution_problem(solution_id)
        if pipeline is None:
            logger.info('method=FitSolution, solution_id=%s, status=ERRORED, error=Solution_id not found', solution_id)
            response = core_pb2.FitSolutionResponse()
            return response

        input_data = [load_data(utils.decode_value(x)) for x in request.inputs]

        expose_outputs = [expose_output for expose_output in request.expose_outputs]
        if expose_outputs:
            expose_outputs = True
        else:
            expose_outputs = False

        request_id = self.backend.fit_pipeline_request(
            problem_description=problem_description, pipeline=pipeline,
            input_data=input_data, expose_outputs=expose_outputs
        )

        response = core_pb2.FitSolutionResponse(request_id=request_id)
        return response

    def GetFitSolutionResults(self, request, context):
        request_id = request.request_id
        logger.info('method=GetFitSolutionResults request_id=%s', request_id)

        progress_start = p_timestamp.Timestamp()
        progress_end = p_timestamp.Timestamp()
        progress_start.GetCurrentTime()

        progress = core_pb2.Progress(state='RUNNING', status='Running fit job', start=progress_start)
        response = core_pb2.GetFitSolutionResultsResponse(progress=progress)
        yield response

        pipeline_result = self.backend.get_request(request_id)
        progress_end.GetCurrentTime()

        if pipeline_result.error is None:
            progress = core_pb2.Progress(
                state='COMPLETED',
                status='Fit job COMPLETED',
                start=progress_start,
                end=progress_end
            )
            response = core_pb2.GetFitSolutionResultsResponse(
                progress=progress, steps=[], exposed_outputs=encode_exposed_values(pipeline_result.exposed_outputs),
                fitted_solution_id=pipeline_result.fitted_pipeline_id
            )
        else:
            progress = core_pb2.Progress(
                state='ERRORED',
                status=str(pipeline_result.error),
                start=progress_start,
                end=progress_end
            )

            response = core_pb2.GetFitSolutionResultsResponse(progress=progress)
        yield response
        return

    def ProduceSolution(self, request, context):
        fitted_solution_id = request.fitted_solution_id
        logger.info('method=ProduceSolution, fitted_solution_id=%s', fitted_solution_id)

        if not self.backend.fitted_pipeline_id_exists(fitted_solution_id):
            logger.info(
                'method=ProduceSolution, fitted_solution_id=%s, status=ERRORED info=No fitted_solution_id found', fitted_solution_id)
            response = core_pb2.ProduceSolutionResponse()
            return response

        input_data = [load_data(utils.decode_value(x)) for x in request.inputs]

        expose_outputs = [expose_output for expose_output in request.expose_outputs]
        if expose_outputs:
            expose_outputs = True
        else:
            expose_outputs = False

        request_id = self.backend.produce_pipeline_request(fitted_pipeline_id=fitted_solution_id,
                                                           input_data=input_data, expose_outputs=expose_outputs)
        response = core_pb2.ProduceSolutionResponse(request_id=request_id)
        return response

    # TODO add expose_outputs to files
    def GetProduceSolutionResults(self, request, context):
        request_id = request.request_id
        logger.info('method=GetProduceSolutionResults, request_id=%s', request_id)

        progress_start = p_timestamp.Timestamp()
        progress_end = p_timestamp.Timestamp()
        progress_start.GetCurrentTime()

        progress = core_pb2.Progress(state='RUNNING', status='Running produce job', start=progress_start)
        response = core_pb2.GetProduceSolutionResultsResponse(progress=progress)
        yield response

        pipeline_result = self.backend.get_request(request_id)
        progress_end.GetCurrentTime()

        if pipeline_result.error is None:
            progress = core_pb2.Progress(
                state='COMPLETED',
                status='Produce job COMPLETED',
                start=progress_start,
                end=progress_end
            )
            step_progress = []

            response = core_pb2.GetProduceSolutionResultsResponse(
                progress=progress, steps=step_progress, exposed_outputs=encode_exposed_values(pipeline_result.exposed_outputs))
        else:
            progress = core_pb2.Progress(
                state='ERRORED',
                status=str(pipeline_result.error),
                start=progress_start,
                end=progress_end
            )

            response = core_pb2.GetProduceSolutionResultsResponse(progress=progress)
        yield response
        return

    def SolutionExport(self, request, context):
        solution_id = request.solution_id
        rank = request.rank

        try:
            pipeline, _, search_id = self.get_solution_problem(solution_id)
        except:
            pipeline = None

        if pipeline is None:
            logger.info('method=SolutionExport, solution_id=%s, status=ERRORED, error=No solution_id found', solution_id)
        else:
            logger.info('method=SolutionExport solution_id=%s', solution_id)
            save_pipeline(pipeline, SearchPath(search_id).pipelines_ranked, rank=rank)
        response = core_pb2.SolutionExportResponse()
        return response

    # def SaveSolution(self, request, context):
    #     solution_id = request.solution_id
    #     logger.info('method=SaveSolution solution_id=%s', solution_id)
    #
    #     if solution_id not in self.manager.solutions:
    #         logger.info('method=SaveSolution, solution_id=%s, error=Solution_id not found', solution_id)
    #         response = core_pb2.SaveSolutionResponse()
    #     else:
    #         solution_uri = self.manager.save_solution(solution_id)
    #         response = core_pb2.SaveSolutionResponse(solution_uri=solution_uri)
    #     return response

    # def LoadSolution(self, request, context):
    #     solution_uri = request.solution_uri
    #     logger.info('method=LoadSolution solution_uri=%s', solution_uri)
    #
    #     if not os.path.exists(solution_uri):
    #         logger.info('method=LoadSolution, solution_uri=%s, error=solution_uri not found', solution_uri)
    #         response = core_pb2.LoadSolutionResponse()
    #     else:
    #         solution_id = self.manager.load_solution(solution_uri)
    #         response = core_pb2.LoadSolutionResponse(solution_id=solution_id)
    #     return response

    # def SaveFittedSolution(self, request, context):
    #     fitted_solution_id = request.fitted_solution_id
    #     logger.info('method=SaveFittedSolution, fitted_solution_id=%s', fitted_solution_id)
    #
    #     if fitted_solution_id not in self.manager.fitted_solutions:
    #         logger.info('method=SaveFittedSolution, fitted_solution_id=%s, status=ERRORED, '
    #                     'info=No fitted_solution_id found', fitted_solution_id)
    #         response = core_pb2.SaveFittedSolutionResponse()
    #     else:
    #         fitted_solution_uri = self.manager.save_fitted_solution(fitted_solution_id)
    #         response = core_pb2.SaveFittedSolutionResponse(fitted_solution_uri=fitted_solution_uri)
    #     return response

    # def LoadFittedSolution(self, request, context):
    #     fitted_solution_uri = request.fitted_solution_uri
    #     logger.info('method=LoadFittedSolution solution_uri=%s', fitted_solution_uri)
    #
    #     if not os.path.exists(fitted_solution_uri):
    #         logger.info('method=LoadFittedSolution, solution_uri=%s, error=solution_uri not found', fitted_solution_uri)
    #         response = core_pb2.LoadFittedSolutionResponse()
    #     else:
    #         fitted_solution_id = self.manager.load_fitted_solution(fitted_solution_uri)
    #         response = core_pb2.LoadFittedSolutionResponse(fitted_solution_id=fitted_solution_id)
    #     return response

    # def ScorePredictions(self, request, context):
    #     logger.info('method=ScorePredictions')
    #     predictions = utils.decode_value(request.predictions)
    #     score_input = utils.decode_value(request.score_input)
    #     problem = utils.decode_problem_description(request.problem)
    #     metrics = [utils.decode_performance_metric(_metric) for _metric in request.metric]
    #
    #     scores, score_result = self.manager.score_predictions(predictions, score_input, problem, metrics)
    #     if score_result.has_error():
    #         logger.info('method=ScorePredictions, error={}', score_result.error)
    #         response = core_pb2.ScorePredictionsResponse()
    #     else:
    #         scores = self.encode_scores(scores)
    #         response = core_pb2.ScorePredictionsResponse(scores=scores)
    #     return response

    def DataAvailable(self, request, context):
        user_agent = request.user_agent
        version = request.version
        time_bound = request.time_bound

        logger.info('method=DataAvailable, agent={}, version={}, time_bound={}'.format(
            user_agent, version, time_bound))
        response = core_pb2.DataAvailableResponse()
        return response

    def SplitData(self, request, context):
        input_data = [load_data(utils.decode_value(x)) for x in request.inputs]
        scoring_configuration = decode_scoring_configuration(request.scoring_configuration)
        problem_description = utils.decode_problem_description(request.problem)
        data_pipeline = schemas_utils.get_splitting_pipeline(scoring_configuration['method'])

        data_random_seed = 0
        outputs, data_result = runtime_module.prepare_data(
            data_pipeline=data_pipeline, problem_description=problem_description,
            inputs=input_data, data_params=scoring_configuration, context=Context.TESTING, random_seed=data_random_seed,
            volumes_dir=EnvVars.D3MSTATICDIR, scratch_dir=Path.TEMP_STORAGE_ROOT, runtime_environment=None,
        )

        if data_result.has_error():
            logger.info('method=SplitData, error={}', data_result.error)
            response = core_pb2.SplitDataResponse()
            yield response
            return
        else:
            for i, (train_output, test_output, score_output) in enumerate(zip(*outputs)):
                uri_list = []
                for output, tag in (
                    (train_output, 'train'),
                    (test_output, 'test'),
                    (score_output, 'score'),
                ):
                    path = os.path.join(
                        Path.TEMP_STORAGE_ROOT, '{}_output_{}'.format(tag, i), 'datasetDoc.json')
                    uri = get_uri(path)
                    output.save(uri)
                    uri_list.append(uri)
                # response
                response = core_pb2.SplitDataResponse(
                    train_output=value_pb2.Value(dataset_uri=uri_list[0]),
                    test_output=value_pb2.Value(dataset_uri=uri_list[1]),
                    score_output=value_pb2.Value(dataset_uri=uri_list[2]),
                )
                yield response

    def ListPrimitives(self, request, context):
        logger.info('method=ListPrimitives')
        primitives_list = []
        for primitive_info in PRIMITIVES_LIST:
            primitives_list.append(primitive_pb2.Primitive(**primitive_info))
        response = core_pb2.ListPrimitivesResponse(primitives=primitives_list)
        return response

    def Hello(self, request, context):
        logger.info('method=Hello')
        user_agent = AGENT
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version]
        allowed_value_types = ALLOWED_VALUE_TYPES
        supported_extensions = SUPPORTED_EXTENSIONS

        response = core_pb2.HelloResponse(
            user_agent=user_agent,
            version=version,
            allowed_value_types=allowed_value_types,
            supported_extensions=supported_extensions
        )
        return response

    def get_solution_problem(self, solution_id):
        describe_search_id = None
        for search_id, solution_ids in self.solutions.items():
            if solution_id in solution_ids:
                describe_search_id = search_id
                break

        if describe_search_id is None:
            return None, None, None

        solution_path = os.path.join(SearchPath(describe_search_id).pipelines_scored, '{}.json'.format(solution_id))

        with d3m_utils.silence():
            pipeline = load_pipeline(solution_path)

        problem_description = self.problem_descriptions[describe_search_id]
        return pipeline, problem_description, describe_search_id


def encode_exposed_values(exposed_values):
    encoded_exposed_values = {}
    for name, value in exposed_values.items():
        if '.csv' in value:
            encoded_exposed_values[name] = utils.encode_value(
                {'type': 'csv_uri', 'value': get_uri(value)}, ALLOWED_VALUE_TYPES, Path.TEMP_STORAGE_ROOT)
        elif '.json' in value:
            encoded_exposed_values[name] = utils.encode_value(
                {'type': 'dataset_uri', 'value': get_uri(value)}, ALLOWED_VALUE_TYPES, Path.TEMP_STORAGE_ROOT)
    return encoded_exposed_values


def decode_scoring_configuration(scoring_configuration):
    """
    Decode a scoring configuration from grpc

    Parameters
    ----------
    scoring_configuration: core_pb2.ScoringConfiguration
        A grpc ScoringConfiguration message.

    Returns
    -------
    configuration: dict
        A dictionary with the scoring configuration.
    """
    method = scoring_configuration.method
    configuration = {
        'method': method,
        'train_score_ratio': str(scoring_configuration.train_test_ratio),
        'stratified': str(scoring_configuration.stratified).lower(),
        'shuffle': str(scoring_configuration.shuffle).lower(),
        'randomSeed': str(scoring_configuration.random_seed),
    }
    if method == 'K_FOLD':
        configuration['number_of_folds'] = str(scoring_configuration.folds)
    return configuration


def load_data(data):
    if data['type'] == 'dataset_uri':
        return container.dataset.get_dataset(data['value'])


def get_uri(path):
    return pathlib.Path(os.path.abspath(path)).as_uri()


def encode_scores(pipeline_result):
    """
    Encode a dict of scores to a GRPC message

    Parameters
    ----------
    pipeline_result
        A pipeline_result instance that contains the scores and rank to be encoded.

    Returns
    -------
    score_message: GRPC
    A GRPC message
    """
    ranking = {
        'metric': 'RANK',
        'value': pipeline_result.rank,
        'randomSeed': 0,
        'fold': 0,
    }

    all_scores = pipeline_result.scores.append(ranking, ignore_index=True)

    scores = list()
    for score in all_scores.to_dict('index').values():
        score['random_seed'] = score['randomSeed']
        try:
            score['metric'] = {'metric': score['metric']}
        except:
            score['metric'] = {'metric': problem_module.PerformanceMetric[score['metric']]}

        scores.append(utils.encode_score(score, ALLOWED_VALUE_TYPES, Path.TEMP_STORAGE_ROOT))
    return scores


def encode_scoring_configuration(scoring_configuration):
    """
    Decode a scoring configuration from grpc

    Parameters
    ----------
    scoring_configuration: dict
        A dictionary with the scoring configuration.

    Returns
    -------
    scoring_configuration: core_pb2.ScoringConfiguration
        A grpc ScoringConfiguration message.
    """
    if scoring_configuration is None:
        return core_pb2.ScoringConfiguration()
    else:
        method = scoring_configuration['method']
        folds = scoring_configuration.get('number_of_folds', None)
        if folds is not None:
            folds = int(folds)
        train_test_ratio = scoring_configuration.get('train_score_ratio', None)
        if train_test_ratio is not None:
            train_test_ratio = float(train_test_ratio)
        shuffle = scoring_configuration.get('shuffle', None)
        if shuffle is not None:
            shuffle = json.loads(shuffle.lower())
        random_seed = scoring_configuration.get('randomSeed', None)
        if random_seed is not None:
            random_seed = int(random_seed)
        stratified = scoring_configuration.get('stratified', None)
        if stratified is not None:
            stratified = json.loads(stratified.lower())
        return core_pb2.ScoringConfiguration(method=method, folds=folds, train_test_ratio=train_test_ratio,
                                             shuffle=shuffle, random_seed=random_seed, stratified=stratified)


class Server:
    def __init__(self, arguments):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.core = Core()

        core_pb2_grpc.add_CoreServicer_to_server(self.core, self.server)
        self.server.add_insecure_port('[::]:45042')

    def start(self):
        self.server.start()

    def stop(self):
        self.server.stop(0)


def configure_parser(parser, *, skip_arguments=()):
    parser.add_argument(
        '-o', '--output-path', type=str, default=os.path.join(os.getcwd(), "output/"),
        help="path where the outputs would be stored"
    )
    parser.add_argument(
        '-v', '--verbose', type=bool, default=True,
        help="Display detailed log"
    )


def main():
    ray.init(webui_host='127.0.0.1')
    # Creating parser
    parser = argparse.ArgumentParser(description="Starts server from command line")
    configure_parser(parser)
    arguments = parser.parse_args()

    # Setup logger
    verbose_format = '%(asctime)s %(levelname)-8s %(processName)-15s [%(filename)s:%(lineno)d] %(message)s'
    concise_format = '%(asctime)s %(levelname)-8s %(message)s'
    log_format = verbose_format if arguments.verbose else concise_format
    logging.basicConfig(format=log_format,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler('{}/d3m.log'.format(Path.TEMP_STORAGE_ROOT), 'w', 'utf-8')],
                        datefmt='%m/%d %H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    warnings.filterwarnings('ignore')

    server = Server(arguments)

    try:
        load_time = time.time()
        server.start()
        with d3m_utils.silence():
            d3m_index.load_all(blocklist=PrimitivesList.BlockList)
        print('Wait for loading workers for', len(d3m_index.search())*0.3)
        time.sleep(len(d3m_index.search())*0.3)
        # time.sleep(5)
        logger.info('---------- Waiting for Requests ----------')
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        logger.info('############ STOPPING SERVICE ############')
        server.stop()


if __name__ == '__main__':
    main()
