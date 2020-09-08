# from __future__ import print_function

import argparse
import os
import pathlib
from pprint import pprint

import grpc
from d3m import utils as d3m_utils, runtime as runtime_module
from d3m.metadata import problem as problem_module
from ta3ta2_api import core_pb2, core_pb2_grpc, value_pb2, utils

from axolotl.utils import pipeline as pipeline_utils
from axolotl.d3m_grpc import constants

# with d3m_utils.silence():
#     d3m_index.load_all(blocklist=constants.PrimitivesList.BLACK_LIST)


# primitives = [
#     'd3m.primitives.datasets.DatasetToDataFrame',
#     'd3m.primitives.data_transformation.denormalize.Common'
# ]
#
# with d3m_utils.silence():
#     for primitive in primitives:
#         d3m_index.get_primitive(primitive)


LENGTH = 60
ALLOWED_VALUE_TYPES = ['DATASET_URI', 'CSV_URI', 'RAW']
FULL_SPECIFIED_PIPELINE_PATH = 'modules/server/test_full_pipeline.json'
PRE_SPECIFIED_PIPELINE_PATH = 'modules/server/test_placeholder.json'


# PRE_SPECIFIED_PIPELINE_PATH = 'modules/server/test_placeholder_pipeline.json'


def hello_request():
    request = core_pb2.HelloRequest()
    return request


def list_primitives_request():
    request = core_pb2.ListPrimitivesRequest()
    return request


def search_solutions_request(test_paths, specified_template=None):
    user_agent = "test_agent"
    version = core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version]

    time_bound = 0.5
    priority = 10
    # allowed_value_types = [value_pb2.ValueType.Value(value) for value in ALLOWED_VALUE_TYPES]

    problem_description = utils.encode_problem_description(
        problem_module.Problem.load(test_paths['TRAIN']['problem'])
    )

    template = None
    if specified_template == 'FULL':
        with d3m_utils.silence():
            pipeline = pipeline_utils.load_pipeline(FULL_SPECIFIED_PIPELINE_PATH)
        template = utils.encode_pipeline_description(pipeline, ALLOWED_VALUE_TYPES, constants.Path.TEMP_STORAGE_ROOT)
    elif specified_template == 'PRE':  # PRE for PREPROCESSING
        pipeline = runtime_module.get_pipeline(PRE_SPECIFIED_PIPELINE_PATH, load_all_primitives=False)
        template = utils.encode_pipeline_description(pipeline, ALLOWED_VALUE_TYPES, constants.Path.TEMP_STORAGE_ROOT)

    inputs = [
        value_pb2.Value(
            dataset_uri=test_paths['TRAIN']['dataset']
        )
    ]

    request = core_pb2.SearchSolutionsRequest(
        user_agent=user_agent,
        version=version,
        time_bound_search=time_bound,
        priority=priority,
        allowed_value_types=ALLOWED_VALUE_TYPES,
        problem=problem_description,
        template=template,
        inputs=inputs
    )
    return request


def get_search_solution_results_request(search_id):
    request = core_pb2.GetSearchSolutionsResultsRequest(search_id=search_id)
    return request


def fit_solution_request(solution_id, test_paths):
    inputs = [
        value_pb2.Value(
            dataset_uri=test_paths['TRAIN']['dataset']
        )
    ]
    expose_outputs = ['outputs.0']
    expose_value_types = ['CSV_URI']
    users = [
        core_pb2.SolutionRunUser(
            id='test_user',
            chosen=True,
            reason='just because'
        )
    ]
    request = core_pb2.FitSolutionRequest(
        solution_id=solution_id,
        inputs=inputs,
        expose_outputs=expose_outputs,
        expose_value_types=expose_value_types,
        users=users
    )
    return request


def get_fit_solution_results_request(request_id):
    request = core_pb2.GetFitSolutionResultsRequest(
        request_id=request_id
    )
    return request


def produce_solution_request(fitted_solution_id, test_paths):
    inputs = [
        value_pb2.Value(
            dataset_uri=test_paths['TEST']['dataset']
        )
    ]
    expose_outputs = ['outputs.0']
    expose_value_types = ['CSV_URI']

    users = [
        core_pb2.SolutionRunUser(
            id='test_user',
            chosen=True,
            reason='just because'
        )
    ]

    request = core_pb2.ProduceSolutionRequest(
        fitted_solution_id=fitted_solution_id,
        inputs=inputs,
        expose_outputs=expose_outputs,
        expose_value_types=expose_value_types,
        users=users
    )
    return request


def get_produce_solution_results_request(request_id):
    request = core_pb2.GetProduceSolutionResultsRequest(
        request_id=request_id
    )
    return request


def describe_solution_request(solution_id):
    request = core_pb2.DescribeSolutionRequest(
        solution_id=solution_id
    )
    return request


def score_solution_request(solution_id, test_paths):
    inputs = [
        value_pb2.Value(
            dataset_uri=test_paths['SCORE']['dataset']
        )
    ]

    problem = problem_module.Problem.load(test_paths['SCORE']['problem'])
    performance_metrics = []
    for performance_metric in problem['problem'].get('performance_metrics', []):
        performance_metrics.append(utils.encode_performance_metric(performance_metric))

    # TODO add support for more evaluation methods
    users = []
    evaluation_method = 'K_FOLD'
    configuration = core_pb2.ScoringConfiguration(
        method=evaluation_method,
        folds=2,
        # train_test_ratio
        shuffle=True,
        random_seed=42,
        stratified=True,
    )
    request = core_pb2.ScoreSolutionRequest(
        solution_id=solution_id,
        inputs=inputs,
        performance_metrics=performance_metrics,
        users=users,
        configuration=configuration
    )
    return request


def get_score_solution_request(solution_id):
    request = core_pb2.ScoreSolutionRequest(
        solution_id=solution_id
    )
    return request


def solution_export_request(solution_id):
    rank = 0.1
    request = core_pb2.SolutionExportRequest(
        solution_id=solution_id,
        rank=rank
    )
    return request


def end_search_solutions_request(search_id):
    request = core_pb2.EndSearchSolutionsRequest(search_id=search_id)
    return request


def stop_search_solution_request(search_id):
    request = core_pb2.StopSearchSolutionsRequest(search_id=search_id)
    return request


def run(test_paths, specified_template=None):
    channel = grpc.insecure_channel('localhost:45042')
    stub = core_pb2_grpc.CoreStub(channel)

    print_name('Hello')
    hello_r = stub.Hello(hello_request())
    pprint(hello_r)

    print_name('ListPrimitive')
    list_primitives_r = stub.ListPrimitives(list_primitives_request())
    for _primitive in list_primitives_r.primitives:
        print_space()
        pprint(_primitive)

    print_name('SearchSolution')
    search_solutions_r = stub.SearchSolutions(search_solutions_request(test_paths, specified_template))
    search_id = search_solutions_r.search_id
    pprint(search_solutions_r)

    print_name('GetSearchSolutionsResults')
    solution_id = None
    for get_search_solution_r in stub.GetSearchSolutionsResults(get_search_solution_results_request(search_id)):
        print_space()
        pprint(get_search_solution_r)
        if get_search_solution_r.solution_id:
            solution_id = get_search_solution_r.solution_id

    print_name('DescribeSolution')
    describe_solution_r = stub.DescribeSolution(describe_solution_request(solution_id))
    pprint(describe_solution_r)

    print_name('FitSolution')
    fit_solution_r = stub.FitSolution(fit_solution_request(solution_id, test_paths))
    fit_request_id = fit_solution_r.request_id
    pprint(fit_solution_r)

    print_name('GetFitSolutionResultsRequest')
    fitted_solution_id = None
    for get_git_solution_results_r in stub.GetFitSolutionResults(get_fit_solution_results_request(fit_request_id)):
        print_space()
        pprint(get_git_solution_results_r)
        fitted_solution_id = get_git_solution_results_r.fitted_solution_id

    print_name('ProduceSolutionRequest')
    produce_solution_r = stub.ProduceSolution(produce_solution_request(fitted_solution_id, test_paths))
    produce_request_id = produce_solution_r.request_id
    pprint(produce_solution_r)

    print_name('GetProduceSolutionResultsRequest')
    for get_produce_solution_results_r in stub.GetProduceSolutionResults(
            get_produce_solution_results_request(produce_request_id)):
        print_space()
        pprint(get_produce_solution_results_r)

    print_name('ScoreSolution')
    score_solution_r = stub.ScoreSolution(score_solution_request(solution_id, test_paths))
    score_request_id = score_solution_r.request_id

    pprint(score_solution_r)

    print_name('GetScoreSolutionResults')
    for score_solution_r in stub.GetScoreSolutionResults(get_score_solution_request(score_request_id)):
        print_space()
        pprint(score_solution_r)

    print_name('SolutionExport')
    solution_export_r = stub.SolutionExport(solution_export_request(solution_id))
    pprint(solution_export_r)

    print_name('StopSearchSolutions')
    stop_search_solution_r = stub.StopSearchSolutions(stop_search_solution_request(search_id))
    pprint(stop_search_solution_r)

    print_name('EndSearchSolutions')
    end_search_solutions_r = stub.EndSearchSolutions(end_search_solutions_request(search_id))
    pprint(end_search_solutions_r)


def print_name(name):
    length = LENGTH
    free_space = length - len(name) - 2
    space = int(free_space / 2)
    name = '#' + ' ' * space + name + ' ' * space
    if free_space % 2 == 0:
        name = name + '#'
    else:
        name = name + ' #'

    print("#" * length)
    print(name)
    print("#" * length)


def print_space():
    print('-' * LENGTH)


def configure_parser(parser, *, skip_arguments=()):
    parser.add_argument(
        '-t', '--test-path', type=str, default="/D3M/internal_d3m/Winter_2018_tamuta2/datasets/26/",
        help="path of d3m dataset to test."
    )


def get_problem_id(test_path):
    problem_description = problem_module.Problem.load(test_path)
    print(problem_description)
    problem_id = problem_description.get('id', None)
    return problem_id


def get_paths(test_path):
    # Classification Score dataset path is (problem_SCORE, dataset_SCORE) not
    # However, regression and other Score dataset path is (problem_TEST, dataset_TEST)
    score_problem_relative_path = os.path.join(test_path, 'SCORE/problem_SCORE/problemDoc.json')
    score_dataset_relative_path = os.path.join(test_path, 'SCORE/dataset_SCORE/datasetDoc.json')

    if not os.path.exists(score_problem_relative_path) or not os.path.exists(score_dataset_relative_path):
        score_problem_relative_path = os.path.join(test_path, 'SCORE/problem_TEST/problemDoc.json')
        score_dataset_relative_path = os.path.join(test_path, 'SCORE/dataset_TEST/datasetDoc.json')

    test_paths = {
        'TRAIN': {
            'dataset': os.path.join(test_path, 'TRAIN/dataset_TRAIN/datasetDoc.json'),
            'problem': pathlib.Path(
                os.path.abspath(os.path.join(test_path, 'TRAIN/problem_TRAIN/problemDoc.json'))).as_uri()
        },
        'TEST': {
            'dataset': os.path.join(test_path, 'TEST/dataset_TEST/datasetDoc.json'),
            'problem': pathlib.Path(
                os.path.abspath(os.path.join(test_path, 'TEST/problem_TEST/problemDoc.json'))).as_uri()
        },
        'SCORE': {
            'dataset': os.path.join(test_path, score_dataset_relative_path),
            'problem': pathlib.Path(os.path.abspath(score_problem_relative_path)).as_uri()
        },
    }
    return test_paths


if __name__ == '__main__':
    # Creating parser
    parser = argparse.ArgumentParser(description="Test from command line")
    configure_parser(parser)
    arguments = parser.parse_args()

    # Getting test root path
    test_path = arguments.test_path

    # Getting test paths train/test/score
    test_paths = get_paths(test_path)

    # Getting problem id
    test_id = get_problem_id(test_paths['TEST']['problem'])

    print_name('Starting Test: ' + test_id)
    run(test_paths, None)
    print_name('Finishing Test: ' + test_id)
