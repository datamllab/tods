import os
import copy
import json
import typing
import logging
import math
import random
import binascii

from d3m import container
from d3m.metadata.problem import TaskKeyword, PerformanceMetric
from d3m.metadata.pipeline import Pipeline
from d3m import utils as d3m_utils

from axolotl.utils import pipeline as pipeline_utils

logger = logging.getLogger(__name__)


# ContainerType = typing.Union[container.Dataset, container.DataFrame, container.ndarray, container.List]
ContainerType = container.Dataset

resource_dir = os.path.dirname(__file__)
SPLITTING_PIPELINES_DIR = os.path.join(resource_dir, 'resources', 'splitting_pipelines.json')
SCORING_PIPELINES_DIR = os.path.join(resource_dir, 'resources', 'scoring_pipeline.yml')
PIPELINES_DB_DIR = os.path.join(resource_dir, 'resources', 'default_pipelines.json')

TASK_TYPE = {
    TaskKeyword.CLASSIFICATION, TaskKeyword.REGRESSION,
    TaskKeyword.CLUSTERING, TaskKeyword.LINK_PREDICTION,
    TaskKeyword.VERTEX_NOMINATION, TaskKeyword.COMMUNITY_DETECTION,
    TaskKeyword.GRAPH_MATCHING, TaskKeyword.COLLABORATIVE_FILTERING,
    TaskKeyword.OBJECT_DETECTION, TaskKeyword.VERTEX_CLASSIFICATION,
    TaskKeyword.FORECASTING
}

TASK_SUBTYPES = {
    TaskKeyword.MULTIVARIATE,
    TaskKeyword.BINARY,
    TaskKeyword.NONOVERLAPPING,
    TaskKeyword.OVERLAPPING,
    TaskKeyword.UNIVARIATE,
    TaskKeyword.MULTICLASS,
    TaskKeyword.MULTILABEL,
}

DATA_TYPES = {
    TaskKeyword.TIME_SERIES,
    TaskKeyword.AUDIO,
    TaskKeyword.TABULAR,
    TaskKeyword.TEXT,
    TaskKeyword.VIDEO,
    TaskKeyword.GRAPH,
    TaskKeyword.IMAGE,
    TaskKeyword.GEOSPATIAL,
    TaskKeyword.RELATIONAL,
    TaskKeyword.GROUPED,
    TaskKeyword.LUPI
}

CLASSIFICATION_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}},
    {'metric': PerformanceMetric.PRECISION, 'params': {}},
    {'metric': PerformanceMetric.RECALL, 'params': {}},
    {'metric': PerformanceMetric.F1, 'params': {}},
    {'metric': PerformanceMetric.F1_MICRO, 'params': {}},
    {'metric': PerformanceMetric.F1_MACRO, 'params': {}},
    {'metric': PerformanceMetric.ROC_AUC, 'params': {}},
]

BINARY_CLASSIFICATION_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}},
]

MULTICLASS_CLASSIFICATION_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}},
    {'metric': PerformanceMetric.F1_MICRO, 'params': {}},
    {'metric': PerformanceMetric.F1_MACRO, 'params': {}},
]

MULTILABEL_CLASSIFICATION_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}},
]

REGRESSION_METRICS = [
    {'metric': PerformanceMetric.MEAN_ABSOLUTE_ERROR, 'params': {}},
    {'metric': PerformanceMetric.MEAN_SQUARED_ERROR, 'params': {}},
    {'metric': PerformanceMetric.ROOT_MEAN_SQUARED_ERROR, 'params': {}},
    {'metric': PerformanceMetric.R_SQUARED, 'params': {}},
]

CLUSTERING_METRICS = [
    {'metric': PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION, 'params': {}},
]

LINK_PREDICTION_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}},
]

VERTEX_NOMINATION_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}},
]

COMMUNITY_DETECTION_METRICS = [
    {'metric': PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION, 'params': {}},
]

GRAPH_CLUSTERING_METRICS = []

GRAPH_MATCHING_METRICS = [
    {'metric': PerformanceMetric.ACCURACY, 'params': {}}
]

TIME_SERIES_FORECASTING_METRICS = REGRESSION_METRICS

COLLABORATIVE_FILTERING_METRICS = REGRESSION_METRICS

OBJECT_DETECTION_METRICS = [
    {'metric': PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION, 'params': {}},
]

MULTICLASS_VERTEX_METRICS = MULTICLASS_CLASSIFICATION_METRICS

SEMI_SUPERVISED_MULTICLASS_CLASSIFICATION_METRICS = MULTICLASS_CLASSIFICATION_METRICS

SEMI_SUPERVISED_REGRESSION_METRICS = REGRESSION_METRICS

DATA_PREPARATION_PARAMS = {
    'k_fold_tabular': {
        'method': 'K_FOLD',
        'number_of_folds': '3',
        'stratified': 'false',
        'shuffle': 'true',
        'randomSeed': '42',
    },

    'holdout': {
        'method': 'HOLDOUT',
        'train_score_ratio': '0.2',
        'shuffle': 'true',
        'stratified': 'true',
        'randomSeed': '42',
    },

    'no_stratified_holdout': {
        'method': 'HOLDOUT',
        'train_score_ratio': '0.2',
        'shuffle': 'true',
        'stratified': 'false',
        'randomSeed': '42',
    },

    'no_split': {
        'method': 'TRAINING_DATA',
        'number_of_folds': '1',
        'stratified': 'true',
        'shuffle': 'true',
        'randomSeed': '42',
    },
}

PROBLEM_DEFINITION = {
    'binary_classification': {
        'performance_metrics': BINARY_CLASSIFICATION_METRICS,
        'task_keywords': [TaskKeyword.CLASSIFICATION, TaskKeyword.BINARY]
    },
    'regression': {
        'performance_metrics': REGRESSION_METRICS,
        'task_keywords': [TaskKeyword.UNIVARIATE, TaskKeyword.REGRESSION]
    }

}


def get_task_description(keywords) -> dict:
    """
    A function that parse the keywords from the problem and map them to
    TaskType, SubTasktype and data type eg. tabular, images, audio, etc

    Parameters
    ----------
    keywords: List[d3m.problem.TaskKeyword]
        List of keywords that comes from d3m problem description

    Returns
    -------
    dict
        {
            task_type: str
            task_subtype: str
            data_types: list
            semi: bool
        }
    """

    task_type = None
    task_subtype = None
    data_types = []
    semi = False
    for keyword in keywords:
        if keyword in TASK_TYPE:
            task_type = keyword.name
        elif keyword in TASK_SUBTYPES:
            task_subtype = keyword.name
        elif keyword in DATA_TYPES:
            data_types.append(keyword.name)
        elif keyword.name == TaskKeyword.SEMISUPERVISED:
            semi = True

    # if data_types is empty we assume is tabular:
    if not data_types:
        data_types.append(TaskKeyword.TABULAR)

    return {'task_type': task_type, 'task_subtype': task_subtype, 'data_types': data_types, 'semi': semi}


def get_metrics_from_task(task_des, perf_metrics=None):
    """
    Provides a dictionary of metrics ready to use for perfromance_metrics

    Parameters
    ----------
    task_des: dict
        A dictionary describe the task
    perf_metrics: dict
        A dictionary specifying the needed performance metric parameters

    Returns
    -------
    performance_metrics: dict
        A dict containing performance metrics.
    """
    # For the case thet the user only want to run a full pipeline
    task_type = task_des['task_type']
    task_subtype = task_des['task_subtype']
    data_types = task_des['data_types']
    if not task_des:
        return None
    if TaskKeyword.CLASSIFICATION == task_type or \
            TaskKeyword.VERTEX_CLASSIFICATION == task_type:
        if task_des['semi']:
            # TODO: Temporary solution to binary semi supervised classification
            metrics = SEMI_SUPERVISED_MULTICLASS_CLASSIFICATION_METRICS
        elif TaskKeyword.BINARY == task_subtype:
            metrics = BINARY_CLASSIFICATION_METRICS
        elif TaskKeyword.MULTICLASS == task_subtype:
            metrics = MULTICLASS_CLASSIFICATION_METRICS
        elif TaskKeyword.MULTILABEL == task_subtype:
            metrics = MULTILABEL_CLASSIFICATION_METRICS
        else:
            metrics = CLASSIFICATION_METRICS
    elif TaskKeyword.REGRESSION == task_type:
        metrics = REGRESSION_METRICS
    elif TaskKeyword.CLUSTERING == task_type:
        metrics = CLUSTERING_METRICS
    elif TaskKeyword.LINK_PREDICTION == task_type:
        metrics = LINK_PREDICTION_METRICS
    elif TaskKeyword.VERTEX_NOMINATION == task_type:
        metrics = VERTEX_NOMINATION_METRICS
    elif TaskKeyword.COMMUNITY_DETECTION == task_type:
        metrics = COMMUNITY_DETECTION_METRICS
    elif TaskKeyword.GRAPH_MATCHING == task_type:
        metrics = GRAPH_MATCHING_METRICS
    elif TaskKeyword.TIME_SERIES in data_types and TaskKeyword.FORECASTING:
        metrics = TIME_SERIES_FORECASTING_METRICS
    elif TaskKeyword.COLLABORATIVE_FILTERING == task_type:
        metrics = COLLABORATIVE_FILTERING_METRICS
    elif TaskKeyword.OBJECT_DETECTION == task_type:
        metrics = OBJECT_DETECTION_METRICS
    else:
        raise ValueError('Task keywords not supported, keywords: {}'.format(task_des))

    for i, metric in enumerate(metrics):
        for perf_metric in perf_metrics:
            if perf_metric['metric'] == metric['metric'] and 'params' in perf_metric:
                copy_metric = copy.deepcopy(metric)
                copy_metric['params']['pos_label'] = perf_metric['params']['pos_label']
                metrics[i] = copy_metric
    logger.info('get_metrics_from_task:metrics: {}'.format(metrics))
    return metrics


def get_eval_configuration(task_type: str, data_types: typing.Sequence, semi: bool) -> typing.Dict:
    """
    Determines which method of evaluation to use, cross_fold, holdout, etc.

    Parameters
    ----------
    task_type: str
        task type
    data_types: list
        data types
    semi: bool
        is it semi-supervised problem

    Returns
    -------
    eval_configuration: dict
        A dict that contains the evaluation method to use.
    """

    # for the case of no problem return None.
    if not task_type:
        return {}

    if semi:
        # Splitting semi may get empty ground truth, which can cause error in sklearn metric.
        return DATA_PREPARATION_PARAMS['no_split']

    if TaskKeyword.CLASSIFICATION == task_type:
        # These data types tend to take up a lot of time to run, so no k_fold.
        if TaskKeyword.AUDIO in data_types or TaskKeyword.VIDEO in data_types \
                or TaskKeyword.IMAGE in data_types:
            return DATA_PREPARATION_PARAMS['holdout']
        else:
            return DATA_PREPARATION_PARAMS['k_fold_tabular']
    elif TaskKeyword.REGRESSION in data_types:
        return DATA_PREPARATION_PARAMS['no_stratified_holdout']
    else:
        return DATA_PREPARATION_PARAMS['no_split']


def get_splitting_pipeline(splitting_name: str) -> Pipeline:
    with open(SPLITTING_PIPELINES_DIR) as file:
        splitting_pipelines = json.load(file)

    if splitting_name in splitting_pipelines:
        return pipeline_utils.load_pipeline(splitting_pipelines[splitting_name])
    else:
        raise ValueError("{} not supported".format(splitting_name))


def get_scoring_pipeline() -> Pipeline:
    with open(SCORING_PIPELINES_DIR, 'r') as pipeline_file:
        with d3m_utils.silence():
            pipeline = Pipeline.from_yaml(pipeline_file)
    return pipeline


def get_pipelines_db():
    with open(PIPELINES_DB_DIR) as file:
        pipelines_dict = json.load(file)
    return pipelines_dict


def get_task_mapping(task: str) -> str:
    """
    Map the task in problem_doc to the task types that are currently supported

    Parameters
    ----------
    task: str
        The task type in problem_doc

    Returns
    -------
    str
        One of task types that are supported

    """
    mapping = {
        'LINK_PREDICTION': 'CLASSIFICATION',
        TaskKeyword.VERTEX_CLASSIFICATION: 'CLASSIFICATION',
        'COMMUNITY_DETECTION': 'CLASSIFICATION',
        'GRAPH_MATCHING': 'CLASSIFICATION',
        TaskKeyword.FORECASTING: 'REGRESSION',
        'OBJECT_DETECTION': 'CLASSIFICATION',
        'VERTEX_CLASSIFICATION': 'CLASSIFICATION',
    }
    if task in mapping:
        return mapping[task]
    else:
        return task



def hex_to_binary(hex_identifier):
    return binascii.unhexlify(hex_identifier)


def binary_to_hex(identifier):
    hex_identifier = binascii.hexlify(identifier)
    return hex_identifier.decode()


def summarize_performance_metrics(performance_metrics):
    """
    A function that averages all the folds if they exist.

    Parameters
    ----------
    performance_metrics: dict
        A dictionary containing the fold, metrics targets and values from evaluation.
    """
    sumarized_performance_metrics = {}

    for metric in performance_metrics.metric.unique():
        mean = performance_metrics[performance_metrics.metric == metric]['value'].mean()
        std = performance_metrics[performance_metrics.metric == metric]['value'].std()
        if math.isnan(std):
            std = 0
        sumarized_performance_metrics[metric] = {
            'mean': mean,
            'std': std,
        }
    return sumarized_performance_metrics


def compute_score(sumarized_performance_metrics):
    """
    A function that computes the internal score based on the average normalized metrics.

    Parameters
    ----------
    sumarized_performance_metrics: dict
     A dictionary containing the summarized version.
    """
    score = 0

    for metric, info in sumarized_performance_metrics.items():
        score += PerformanceMetric[metric].normalize(info['mean'])

    score = score / float(len(sumarized_performance_metrics))
    return score


def compute_rank(sumarized_performance_metrics):
    """
    A function that computes the rank based on the average normalized metrics.

    Parameters
    ----------
    sumarized_performance_metrics: dict
     A dictionary containing the summarized version.
    """
    ranks = {}
    mean = 0
    for metric, info in sumarized_performance_metrics.items():
        try:
            ranks[metric] = PerformanceMetric[metric].normalize(abs(info['mean'] - info['std']))
        except:
            ranks[metric] = 0
        mean += ranks[metric]

    mean = mean / len(sumarized_performance_metrics)
    # rank = 1 - ranks[min(ranks.keys(), key=(lambda k: ranks[k]))] + random.randint(10, 30)**-6
    rank = 1 - mean

    # We add some randomness on the rank to avoid duplications
    noise = 0
    sign = -1 if random.randint(0, 1) == 0 else 1
    range_0 = -9
    range_1 = -5
    if rank < 1e-5:
        range_0 = -12
        range_1 = -9

    for i in range(range_0, range_1):
        noise += random.randint(0, 9) * 10 ** i
    rank = rank + noise * sign
    if rank < 0:
        rank *= -1
    return rank


def random_rank():
    ranks = 0
    average_number = 5
    for i in range(average_number):
        ranks += random.uniform(0, 1)
    ranks = ranks/average_number
    return ranks
