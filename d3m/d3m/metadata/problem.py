import abc
import argparse
import copy
import functools
import json
import logging
import math
import os.path
import pprint
import sys
import traceback
import typing
from urllib import parse as url_parse

from . import base
from d3m import deprecate, exceptions, utils

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import lib as pyarrow_lib  # type: ignore
except ModuleNotFoundError:
    pyarrow_lib = None

__all__ = ('TaskKeyword', 'PerformanceMetric', 'Problem')

logger = logging.getLogger(__name__)

# Comma because we unpack the list of validators returned from "load_schema_validators".
PROBLEM_SCHEMA_VALIDATOR, = utils.load_schema_validators(base.SCHEMAS, ('problem.json',))

PROBLEM_SCHEMA_VERSION = 'https://metadata.datadrivendiscovery.org/schemas/v0/problem.json'


def sigmoid(x: float) -> float:
    """
    Numerically stable scaled logistic function.

    Maps all values ``x`` to [0, 1]. Values between -1000 and 1000 are
    mapped reasonably far from 0 and 1, after which the function
    converges to bounds.

    Parameters
    ----------
    x:
        Input.

    Returns
    -------
    Output.
    """

    scale = 1 / 1000

    if x >= 0:
        ex = math.exp(scale * -x)
        return 1 / (1 + ex)
    else:
        ex = math.exp(scale * x)
        return ex / (1 + ex)


class TaskKeywordBase:
    _d3m_map: typing.Dict[str, 'TaskKeywordBase'] = {}

    @classmethod
    def get_map(cls) -> dict:
        """
        Returns the map between D3M problem description JSON string and enum values.

        Returns
        -------
        The map.
        """

        return cls._d3m_map

    @classmethod
    def parse(cls, name: str) -> 'TaskKeywordBase':
        """
        Converts D3M problem description JSON string into enum value.

        Parameters
        ----------
        name:
            D3M problem description JSON string.

        Returns
        -------
        Enum value.
        """

        return cls.get_map()[name]

    def unparse(self) -> str:
        """
        Converts enum value to D3M problem description JSON string.

        Returns
        -------
        D3M problem description JSON string.
        """

        for key, value in self.get_map().items():
            if self == value:
                return key

        raise exceptions.InvalidStateError("Cannot convert {self}.".format(self=self))

    def __ge__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) >= list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented

    def __gt__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) > list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented

    def __le__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) <= list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented

    def __lt__(self, other: typing.Any) -> bool:
        if self.__class__ is other.__class__:
            return list(self.__class__.__members__.keys()).index(self.value) < list(other.__class__.__members__.keys()).index(other.value)  # type: ignore
        return NotImplemented


TaskKeyword = utils.create_enum_from_json_schema_enum(
    'TaskKeyword', base.DEFINITIONS_JSON,
    'definitions.problem.properties.task_keywords.items.oneOf[*].enum[*]',
    module=__name__, base_class=TaskKeywordBase,
)

TaskKeyword._d3m_map.update({
    'classification': TaskKeyword.CLASSIFICATION,  # type: ignore
    'regression': TaskKeyword.REGRESSION,  # type: ignore
    'clustering': TaskKeyword.CLUSTERING,  # type: ignore
    'linkPrediction': TaskKeyword.LINK_PREDICTION,  # type: ignore
    'vertexNomination': TaskKeyword.VERTEX_NOMINATION,  # type: ignore
    'vertexClassification': TaskKeyword.VERTEX_CLASSIFICATION,  # type: ignore
    'communityDetection': TaskKeyword.COMMUNITY_DETECTION,  # type: ignore
    'graphMatching': TaskKeyword.GRAPH_MATCHING,  # type: ignore
    'forecasting': TaskKeyword.FORECASTING,  # type: ignore
    'collaborativeFiltering': TaskKeyword.COLLABORATIVE_FILTERING,  # type: ignore
    'objectDetection': TaskKeyword.OBJECT_DETECTION,  # type: ignore
    'semiSupervised': TaskKeyword.SEMISUPERVISED,  # type: ignore
    'binary': TaskKeyword.BINARY,  # type: ignore
    'multiClass': TaskKeyword.MULTICLASS,  # type: ignore
    'multiLabel': TaskKeyword.MULTILABEL,  # type: ignore
    'univariate': TaskKeyword.UNIVARIATE,  # type: ignore
    'multivariate': TaskKeyword.MULTIVARIATE,  # type: ignore
    'overlapping': TaskKeyword.OVERLAPPING,  # type: ignore
    'nonOverlapping': TaskKeyword.NONOVERLAPPING,  # type: ignore
    'tabular': TaskKeyword.TABULAR,  # type: ignore
    'relational': TaskKeyword.RELATIONAL,  # type: ignore
    'nested': TaskKeyword.NESTED,  # type: ignore
    'image': TaskKeyword.IMAGE,  # type: ignore
    'audio': TaskKeyword.AUDIO,  # type: ignore
    'video': TaskKeyword.VIDEO,  # type: ignore
    'speech': TaskKeyword.SPEECH,  # type: ignore
    'text': TaskKeyword.TEXT,  # type: ignore
    'graph': TaskKeyword.GRAPH,  # type: ignore
    'multiGraph': TaskKeyword.MULTIGRAPH,  # type: ignore
    'timeSeries': TaskKeyword.TIME_SERIES,  # type: ignore
    'grouped': TaskKeyword.GROUPED,  # type: ignore
    'geospatial': TaskKeyword.GEOSPATIAL,  # type: ignore
    'remoteSensing': TaskKeyword.REMOTE_SENSING,  # type: ignore
    'lupi': TaskKeyword.LUPI,  # type: ignore
    'missingMetadata': TaskKeyword.MISSING_METADATA,  # type: ignore
})


class PerformanceMetricBase:
    _d3m_map: typing.ClassVar[typing.Dict[str, 'PerformanceMetricBase']] = {}
    _requires_confidence_set: typing.ClassVar[typing.Set['PerformanceMetricBase']] = set()
    _requires_rank_set: typing.ClassVar[typing.Set['PerformanceMetricBase']] = set()
    _best_value_map: typing.ClassVar[typing.Dict['PerformanceMetricBase', float]] = {}
    _worst_value_map: typing.ClassVar[typing.Dict['PerformanceMetricBase', float]] = {}
    _additional_score_class_map: typing.ClassVar[typing.Dict['PerformanceMetricBase', type]] = {}

    @classmethod
    def get_map(cls) -> dict:
        """
        Returns the map between D3M problem description JSON string and enum values.

        Returns
        -------
        The map.
        """

        return cls._d3m_map

    @classmethod
    def parse(cls, name: str) -> 'PerformanceMetricBase':
        """
        Converts D3M problem description JSON string into enum value.

        Parameters
        ----------
        name:
            D3M problem description JSON string.

        Returns
        -------
        Enum value.
        """

        return cls.get_map()[name]

    def unparse(self) -> str:
        """
        Converts enum value to D3M problem description JSON string.

        Returns
        -------
        D3M problem description JSON string.
        """

        for key, value in self.get_map().items():
            if self == value:
                return key

        raise exceptions.InvalidStateError("Cannot convert {self}.".format(self=self))

    def requires_confidence(self) -> bool:
        """
        Returns ``True`` if this metric requires confidence column.

        Returns
        -------
        ``True`` if this metric requires confidence column.
        """

        return self in self._requires_confidence_set

    def requires_rank(self) -> bool:
        """
        Returns ``True`` if this metric requires rank column.

        Returns
        -------
        ``True`` if this metric requires rank column.
        """

        return self in self._requires_rank_set

    def best_value(self) -> float:
        """
        The best possible value of the metric.

        Returns
        -------
        The best possible value of the metric.
        """

        return self._best_value_map[self]

    def worst_value(self) -> float:
        """
        The worst possible value of the metric.

        Returns
        -------
        The worst possible value of the metric.
        """

        return self._worst_value_map[self]

    def normalize(self, value: float) -> float:
        """
        Normalize the ``value`` for this metric so that it is between 0 and 1,
        inclusive, where 1 is the best score and 0 is the worst.

        Parameters
        ----------
        value:
            Value of this metric to normalize.

        Returns
        -------
        A normalized metric.
        """

        worst_value = self.worst_value()
        best_value = self.best_value()

        return self._normalize(worst_value, best_value, value)

    @classmethod
    def _normalize(cls, worst_value: float, best_value: float, value: float) -> float:
        assert worst_value <= value <= best_value or worst_value >= value >= best_value, (worst_value, value, best_value)

        if math.isinf(best_value) and math.isinf(worst_value):
            value = sigmoid(value)
            if best_value > worst_value:  # "best_value" == inf, "worst_value" == -inf
                best_value = 1.0
                worst_value = 0.0
            else:  # "best_value" == -inf, "worst_value" == inf
                best_value = 0.0
                worst_value = 1.0
        elif math.isinf(best_value):
            value = sigmoid(value - worst_value)
            if best_value > worst_value:  # "best_value" == inf
                best_value = 1.0
                worst_value = 0.5
            else:  # "best_value" == -inf
                best_value = 0.0
                worst_value = 0.5
        elif math.isinf(worst_value):
            value = sigmoid(best_value - value)
            if best_value > worst_value:  # "worst_value" == -inf
                best_value = 0.5
                worst_value = 1.0
            else:  # "worst_value" == inf
                best_value = 0.5
                worst_value = 0.0

        return (value - worst_value) / (best_value - worst_value)

    def get_class(self) -> typing.Any:
        """
        Returns a class suitable for computing this metric.
        """

        # Importing here to prevent import cycle.
        from d3m import metrics

        if self in metrics.class_map:
            return metrics.class_map[self]  # type: ignore

        if self in self._additional_score_class_map:
            return self._additional_score_class_map[self]  # type: ignore

        raise exceptions.NotSupportedError("Computing metric {metric} is not supported.".format(metric=self))

    @classmethod
    def register_metric(cls, name: str, *, best_value: float, worst_value: float, score_class: type, requires_confidence: bool = False, requires_rank: bool = False) -> None:
        cls.register_value(name, name)  # type: ignore
        cls._best_value_map[cls[name]] = best_value  # type: ignore
        cls._worst_value_map[cls[name]] = worst_value  # type: ignore
        cls._additional_score_class_map[cls[name]] = score_class  # type: ignore

        if requires_confidence:
            PerformanceMetric._requires_confidence_set.add(cls[name])  # type: ignore

        if requires_rank:
            PerformanceMetric._requires_rank_set.add(cls[name])  # type: ignore


PerformanceMetric = utils.create_enum_from_json_schema_enum(
    'PerformanceMetric', base.DEFINITIONS_JSON,
    'definitions.performance_metric.oneOf[*].properties.metric.enum[*]',
    module=__name__, base_class=PerformanceMetricBase,
)

PerformanceMetric._d3m_map.update({
    'accuracy': PerformanceMetric.ACCURACY,  # type: ignore
    'precision': PerformanceMetric.PRECISION,  # type: ignore
    'recall': PerformanceMetric.RECALL,  # type: ignore
    'f1': PerformanceMetric.F1,  # type: ignore
    'f1Micro': PerformanceMetric.F1_MICRO,  # type: ignore
    'f1Macro': PerformanceMetric.F1_MACRO,  # type: ignore
    'rocAuc': PerformanceMetric.ROC_AUC,  # type: ignore
    'rocAucMicro': PerformanceMetric.ROC_AUC_MICRO,  # type: ignore
    'rocAucMacro': PerformanceMetric.ROC_AUC_MACRO,  # type: ignore
    'meanSquaredError': PerformanceMetric.MEAN_SQUARED_ERROR,  # type: ignore
    'rootMeanSquaredError': PerformanceMetric.ROOT_MEAN_SQUARED_ERROR,  # type: ignore
    'meanAbsoluteError': PerformanceMetric.MEAN_ABSOLUTE_ERROR,  # type: ignore
    'rSquared': PerformanceMetric.R_SQUARED,  # type: ignore
    'normalizedMutualInformation': PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION,  # type: ignore
    'jaccardSimilarityScore': PerformanceMetric.JACCARD_SIMILARITY_SCORE,  # type: ignore
    'precisionAtTopK': PerformanceMetric.PRECISION_AT_TOP_K,  # type: ignore
    'objectDetectionAP': PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION,  # type: ignore
    'hammingLoss': PerformanceMetric.HAMMING_LOSS,  # type: ignore
    'meanReciprocalRank': PerformanceMetric.MEAN_RECIPROCAL_RANK,  # type: ignore
    'hitsAtK': PerformanceMetric.HITS_AT_K,  # type: ignore
})
PerformanceMetric._requires_confidence_set.update({
    PerformanceMetric.ROC_AUC,
    PerformanceMetric.ROC_AUC_MICRO,
    PerformanceMetric.ROC_AUC_MACRO,
    PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION,
})
PerformanceMetric._requires_rank_set.update({
    PerformanceMetric.MEAN_RECIPROCAL_RANK,
    PerformanceMetric.HITS_AT_K,
})
PerformanceMetric._best_value_map.update({
    PerformanceMetric.ACCURACY: 1.0,  # type: ignore
    PerformanceMetric.PRECISION: 1.0,  # type: ignore
    PerformanceMetric.RECALL: 1.0,  # type: ignore
    PerformanceMetric.F1: 1.0,  # type: ignore
    PerformanceMetric.F1_MICRO: 1.0,  # type: ignore
    PerformanceMetric.F1_MACRO: 1.0,  # type: ignore
    PerformanceMetric.ROC_AUC: 1.0,  # type: ignore
    PerformanceMetric.ROC_AUC_MICRO: 1.0,  # type: ignore
    PerformanceMetric.ROC_AUC_MACRO: 1.0,  # type: ignore
    PerformanceMetric.MEAN_SQUARED_ERROR: 0.0,  # type: ignore
    PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: 0.0,  # type: ignore
    PerformanceMetric.MEAN_ABSOLUTE_ERROR: 0.0,  # type: ignore
    PerformanceMetric.R_SQUARED: 1.0,  # type: ignore
    PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: 1.0,  # type: ignore
    PerformanceMetric.JACCARD_SIMILARITY_SCORE: 1.0,  # type: ignore
    PerformanceMetric.PRECISION_AT_TOP_K: 1.0,  # type: ignore
    PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: 1.0,  # type: ignore
    PerformanceMetric.HAMMING_LOSS: 0.0,  # type: ignore
    PerformanceMetric.MEAN_RECIPROCAL_RANK: 1.0,  # type: ignore
    PerformanceMetric.HITS_AT_K: 1.0,  # type: ignore
})
PerformanceMetric._worst_value_map.update({
    PerformanceMetric.ACCURACY: 0.0,  # type: ignore
    PerformanceMetric.PRECISION: 0.0,  # type: ignore
    PerformanceMetric.RECALL: 0.0,  # type: ignore
    PerformanceMetric.F1: 0.0,  # type: ignore
    PerformanceMetric.F1_MICRO: 0.0,  # type: ignore
    PerformanceMetric.F1_MACRO: 0.0,  # type: ignore
    PerformanceMetric.ROC_AUC: 0.0,  # type: ignore
    PerformanceMetric.ROC_AUC_MICRO: 0.0,  # type: ignore
    PerformanceMetric.ROC_AUC_MACRO: 0.0,  # type: ignore
    PerformanceMetric.MEAN_SQUARED_ERROR: float('inf'),  # type: ignore
    PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: float('inf'),  # type: ignore
    PerformanceMetric.MEAN_ABSOLUTE_ERROR: float('inf'),  # type: ignore
    PerformanceMetric.R_SQUARED: float('-inf'),  # type: ignore
    PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: 0.0,  # type: ignore
    PerformanceMetric.JACCARD_SIMILARITY_SCORE: 0.0,  # type: ignore
    PerformanceMetric.PRECISION_AT_TOP_K: 0.0,  # type: ignore
    PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: 0.0,  # type: ignore
    PerformanceMetric.HAMMING_LOSS: 1.0,  # type: ignore
    PerformanceMetric.MEAN_RECIPROCAL_RANK: 0.0,  # type: ignore
    PerformanceMetric.HITS_AT_K: 0.0,  # type: ignore
})

# Here are all legacy (before v4.0.0) task types and task subtypes mapped to task keywords.
TASK_TYPE_TO_KEYWORDS_MAP: typing.Dict[typing.Optional[str], typing.List] = {
    None: [],
    'classification': [TaskKeyword.CLASSIFICATION],  # type: ignore
    'regression': [TaskKeyword.REGRESSION],  # type: ignore
    'clustering': [TaskKeyword.CLUSTERING],  # type: ignore
    'linkPrediction': [TaskKeyword.LINK_PREDICTION],  # type: ignore
    'vertexClassification': [TaskKeyword.VERTEX_CLASSIFICATION],  # type: ignore
    'vertexNomination': [TaskKeyword.VERTEX_NOMINATION],  # type: ignore
    'communityDetection': [TaskKeyword.COMMUNITY_DETECTION],  # type: ignore
    'graphMatching': [TaskKeyword.GRAPH_MATCHING],  # type: ignore
    'timeSeriesForecasting': [TaskKeyword.TIME_SERIES, TaskKeyword.FORECASTING],  # type: ignore
    'collaborativeFiltering': [TaskKeyword.COLLABORATIVE_FILTERING],  # type: ignore
    'objectDetection': [TaskKeyword.OBJECT_DETECTION],  # type: ignore
    'semiSupervisedClassification': [TaskKeyword.SEMISUPERVISED, TaskKeyword.CLASSIFICATION],  # type: ignore
    'semiSupervisedRegression': [TaskKeyword.SEMISUPERVISED, TaskKeyword.REGRESSION],  # type: ignore
    'binary': [TaskKeyword.BINARY],  # type: ignore
    'multiClass': [TaskKeyword.MULTICLASS],  # type: ignore
    'multiLabel': [TaskKeyword.MULTILABEL],  # type: ignore
    'univariate': [TaskKeyword.UNIVARIATE],  # type: ignore
    'multivariate': [TaskKeyword.MULTIVARIATE],  # type: ignore
    'overlapping': [TaskKeyword.OVERLAPPING],  # type: ignore
    'nonOverlapping': [TaskKeyword.NONOVERLAPPING],  # type: ignore
}
JSON_TASK_TYPE_TO_KEYWORDS_MAP: typing.Dict[typing.Optional[str], typing.List] = {
    None: [],
    'CLASSIFICATION': [TaskKeyword.CLASSIFICATION],  # type: ignore
    'REGRESSION': [TaskKeyword.REGRESSION],  # type: ignore
    'CLUSTERING': [TaskKeyword.CLUSTERING],  # type: ignore
    'LINK_PREDICTION': [TaskKeyword.LINK_PREDICTION],  # type: ignore
    'VERTEX_CLASSIFICATION': [TaskKeyword.VERTEX_CLASSIFICATION],  # type: ignore
    'VERTEX_NOMINATION': [TaskKeyword.VERTEX_NOMINATION],  # type: ignore
    'COMMUNITY_DETECTION': [TaskKeyword.COMMUNITY_DETECTION],  # type: ignore
    'GRAPH_MATCHING': [TaskKeyword.GRAPH_MATCHING],  # type: ignore
    'TIME_SERIES_FORECASTING': [TaskKeyword.TIME_SERIES, TaskKeyword.FORECASTING],  # type: ignore
    'COLLABORATIVE_FILTERING': [TaskKeyword.COLLABORATIVE_FILTERING],  # type: ignore
    'OBJECT_DETECTION': [TaskKeyword.OBJECT_DETECTION],  # type: ignore
    'SEMISUPERVISED_CLASSIFICATION': [TaskKeyword.SEMISUPERVISED, TaskKeyword.CLASSIFICATION],  # type: ignore
    'SEMISUPERVISED_REGRESSION': [TaskKeyword.SEMISUPERVISED, TaskKeyword.REGRESSION],  # type: ignore
    'BINARY': [TaskKeyword.BINARY],  # type: ignore
    'MULTICLASS': [TaskKeyword.MULTICLASS],  # type: ignore
    'MULTILABEL': [TaskKeyword.MULTILABEL],  # type: ignore
    'UNIVARIATE': [TaskKeyword.UNIVARIATE],  # type: ignore
    'MULTIVARIATE': [TaskKeyword.MULTIVARIATE],  # type: ignore
    'OVERLAPPING': [TaskKeyword.OVERLAPPING],  # type: ignore
    'NONOVERLAPPING': [TaskKeyword.NONOVERLAPPING],  # type: ignore
}


class Loader(metaclass=utils.AbstractMetaclass):
    """
    A base class for problem loaders.
    """

    @abc.abstractmethod
    def can_load(self, problem_uri: str) -> bool:
        """
        Return ``True`` if this loader can load a problem from a given URI ``problem_uri``.

        Parameters
        ----------
        problem_uri:
            A URI to load a problem from.

        Returns
        -------
        ``True`` if this loader can load a problem from ``problem_uri``.
        """

    @abc.abstractmethod
    def load(self, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        """
        Loads the problem at ``problem_uri``.

        Parameters
        ----------
        problem_uri:
            A URI to load.
        problem_id:
            Override problem ID determined by the loader.
        problem_version:
            Override problem version determined by the loader.
        problem_name:
            Override problem name determined by the loader.
        strict_digest:
            If computed digest does not match the one provided in metadata, raise an exception?
        handle_score_split:
            Rename a scoring problem to not have the same name as testing problem
            and update dataset references.

        Returns
        -------
        A loaded problem.
        """

    @classmethod
    def get_problem_class(cls) -> 'typing.Type[Problem]':
        return Problem


class D3MProblemLoader(Loader):
    """
    A class for loading of D3M problems.

    Loader support only loading from a local file system.
    URI should point to the ``problemDoc.json`` file in the D3M problem directory.
    """

    SUPPORTED_VERSIONS = {'3.0', '3.1', '3.1.1', '3.1.2', '3.2.0', '3.2.1', '3.3.0', '3.3.1', '4.0.0', '4.1.0'}

    def can_load(self, dataset_uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(dataset_uri, allow_fragments=False)
        except Exception:
            return False

        if parsed_uri.scheme != 'file':
            return False

        if parsed_uri.netloc not in ['', 'localhost']:
            return False

        if not parsed_uri.path.startswith('/'):
            return False

        if os.path.basename(parsed_uri.path) != 'problemDoc.json':
            return False

        return True

    # "strict_digest" is not used because there is no digest in D3M problem descriptions.
    def load(self, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        assert self.can_load(problem_uri)

        parsed_uri = url_parse.urlparse(problem_uri, allow_fragments=False)

        problem_doc_path = parsed_uri.path

        try:
            with open(problem_doc_path, 'r', encoding='utf8') as problem_doc_file:
                problem_doc = json.load(problem_doc_file)
        except FileNotFoundError as error:
            raise exceptions.ProblemNotFoundError(
                "D3M problem '{problem_uri}' cannot be found.".format(problem_uri=problem_uri),
            ) from error

        problem_schema_version = problem_doc.get('about', {}).get('problemSchemaVersion', '3.3.0')
        if problem_schema_version not in self.SUPPORTED_VERSIONS:
            logger.warning("Loading a problem with unsupported schema version '%(version)s'. Supported versions: %(supported_versions)s", {
                'version': problem_schema_version,
                'supported_versions': self.SUPPORTED_VERSIONS,
            })

        # To be compatible with problem descriptions which do not adhere to the schema and have only one entry for data.
        if not isinstance(problem_doc['inputs']['data'], list):
            problem_doc['inputs']['data'] = [problem_doc['inputs']['data']]

        performance_metrics = []
        for performance_metric in problem_doc['inputs']['performanceMetrics']:
            params = {}

            if 'posLabel' in performance_metric:
                params['pos_label'] = performance_metric['posLabel']

            if 'K' in performance_metric:
                params['k'] = performance_metric['K']

            performance_metrics.append({
                'metric': PerformanceMetric.parse(performance_metric['metric']),
            })

            if params:
                performance_metrics[-1]['params'] = params

        inputs = []
        for data in problem_doc['inputs']['data']:
            targets = []
            for target in data['targets']:
                targets.append({
                    'target_index': target['targetIndex'],
                    'resource_id': target['resID'],
                    'column_index': target['colIndex'],
                    'column_name': target['colName'],
                })

                if 'numClusters' in target:
                    targets[-1]['clusters_number'] = target['numClusters']

            privileged_data_columns = []
            for privileged_data in data.get('privilegedData', []):
                privileged_data_columns.append({
                    'privileged_data_index': privileged_data['privilegedDataIndex'],
                    'resource_id': privileged_data['resID'],
                    'column_index': privileged_data['colIndex'],
                    'column_name': privileged_data['colName'],
                })

            problem_input = {
                'dataset_id': data['datasetID'],
            }

            if targets:
                problem_input['targets'] = targets

            if privileged_data_columns:
                problem_input['privileged_data'] = privileged_data_columns

            if data.get('forecastingHorizon', {}).get('horizonValue', None):
                problem_input['forecasting_horizon'] = {
                    'resource_id': data['forecastingHorizon']['resID'],
                    'column_index': data['forecastingHorizon']['colIndex'],
                    'column_name': data['forecastingHorizon']['colName'],
                    'horizon_value': data['forecastingHorizon']['horizonValue'],
                }

            inputs.append(problem_input)

        document_problem_id = problem_doc['about']['problemID']
        # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
        # They are the same as TEST dataset splits, but we present them differently, so that
        # SCORE dataset splits have targets as part of data. Because of this we also update
        # corresponding problem ID.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
        if handle_score_split and os.path.exists(os.path.join(os.path.dirname(problem_doc_path), '..', 'targets.csv')) and document_problem_id.endswith('_TEST'):
            document_problem_id = document_problem_id[:-5] + '_SCORE'

            # Also update dataset references.
            for data in problem_doc.get('inputs', {}).get('data', []):
                if data['datasetID'].endswith('_TEST'):
                    data['datasetID'] = data['datasetID'][:-5] + '_SCORE'

        # "dataSplits" is not exposed as a problem description. One should provide splitting
        # configuration to a splitting pipeline instead. Similarly, "outputs" are not exposed either.
        description = {
            'schema': PROBLEM_SCHEMA_VERSION,
            'id': problem_id or document_problem_id,
            'version': problem_version or problem_doc['about'].get('problemVersion', '1.0'),
            'name': problem_name or problem_doc['about']['problemName'],
            'location_uris': [
                # We reconstruct the URI to normalize it.
                utils.fix_uri(problem_doc_path),
            ],
            'problem': {},
        }

        task_keywords: typing.List = []

        # Legacy (before v4.0.0).
        task_keywords += TASK_TYPE_TO_KEYWORDS_MAP[problem_doc['about'].get('taskType', None)]
        task_keywords += TASK_TYPE_TO_KEYWORDS_MAP[problem_doc['about'].get('taskSubType', None)]

        if problem_doc['about'].get('taskKeywords', []):
            for task_keyword in problem_doc['about']['taskKeywords']:
                task_keywords.append(TaskKeyword.parse(task_keyword))

        if task_keywords:
            description['problem']['task_keywords'] = sorted(set(task_keywords))  # type: ignore

        if performance_metrics:
            description['problem']['performance_metrics'] = performance_metrics  # type: ignore

        if problem_doc['about'].get('problemDescription', None):
            description['description'] = problem_doc['about']['problemDescription']  # type: ignore

        if problem_doc['about'].get('problemURI', None):
            typing.cast(typing.List[str], description['location_uris']).append(problem_doc['about']['problemURI'])

        if inputs:
            description['inputs'] = inputs  # type: ignore

        if 'dataAugmentation' in problem_doc:
            description['data_augmentation'] = problem_doc['dataAugmentation']

        # We do not want empty objects.
        if not description['problem']:
            del description['problem']

        problem_class = self.get_problem_class()

        return problem_class(description)


P = typing.TypeVar('P', bound='Problem')


# TODO: It should be probably immutable.
class Problem(dict):
    """
    A class representing a problem.
    """

    def __init__(self, problem_description: typing.Dict = None, *, strict_digest: bool = False) -> None:
        super().__init__(problem_description)

        PROBLEM_SCHEMA_VALIDATOR.validate(self)

        if 'digest' in self:
            digest = self.get_digest()

            if digest != self['digest']:
                if strict_digest:
                    raise exceptions.DigestMismatchError(
                        "Digest for problem description '{problem_id}' does not match a computed one. Provided digest: {problem_digest}. Computed digest: {new_problem_digest}.".format(
                            problem_id=self['id'],
                            problem_digest=self['digest'],
                            new_problem_digest=digest,
                        )
                    )
                else:
                    logger.warning(
                        "Digest for problem description '%(problem_id)s' does not match a computed one. Provided digest: %(problem_digest)s. Computed digest: %(new_problem_digest)s.",
                        {
                            'problem_id': self['id'],
                            'problem_digest': self['digest'],
                            'new_problem_digest': digest,
                        },
                    )

            # We do not want it to be stored in the object because it can become
            # obsolete. Use "get_digest" to get the current digest.
            del self['digest']

    loaders: typing.List[Loader] = [
        D3MProblemLoader(),
    ]

    @classmethod
    def load(cls, problem_uri: str, *, problem_id: str = None, problem_version: str = None,
             problem_name: str = None, strict_digest: bool = False, handle_score_split: bool = True) -> 'Problem':
        """
        Tries to load problem from ``problem_uri`` using all registered problem loaders.

        Parameters
        ----------
        problem_uri:
            A URI to load.
        problem_id:
            Override problem ID determined by the loader.
        problem_version:
            Override problem version determined by the loader.
        problem_name:
            Override problem name determined by the loader.
        strict_digest:
            If computed digest does not match the one provided in metadata, raise an exception?
        handle_score_split:
            Rename a scoring problem to not have the same name as testing problem
            and update dataset references.

        Returns
        -------
        A loaded problem.
        """

        for loader in cls.loaders:
            if loader.can_load(problem_uri):
                return loader.load(
                    problem_uri, problem_id=problem_id, problem_version=problem_version,
                    problem_name=problem_name, strict_digest=strict_digest,
                    handle_score_split=handle_score_split,
                )

        raise exceptions.ProblemUriNotSupportedError(
            "No known loader could load problem from '{problem_uri}'.".format(problem_uri=problem_uri)
        )

    # TODO: Allow one to specify priority which would then insert loader at a different place and not at the end?
    @classmethod
    def register_loader(cls, loader: Loader) -> None:
        """
        Registers a new problem loader.

        Parameters
        ----------
        loader:
            An instance of the loader class implementing a new loader.
        """

        cls.loaders.append(loader)

    def __repr__(self) -> str:
        return self.__str__()

    def _get_description_keys(self) -> typing.Sequence[str]:
        return 'id', 'name', 'location_uris'

    def __str__(self) -> str:
        return '{class_name}({description})'.format(
            class_name=type(self).__name__,
            description=', '.join('{key}=\'{value}\''.format(key=key, value=self[key]) for key in self._get_description_keys() if key in self),
        )

    def copy(self: P) -> P:
        return copy.deepcopy(self)

    @classmethod
    def _canonical_problem_description(cls: typing.Type[P], problem_description: typing.Dict) -> P:
        """
        Before we compute digest of the problem description, we have to convert it to a
        canonical structure.

        Currently, this is just removing any local URIs the description might have.
        """

        # Making a copy.
        problem_description = dict(problem_description)

        utils.filter_local_location_uris(problem_description)

        if 'digest' in problem_description:
            del problem_description['digest']

        return cls(problem_description)

    def get_digest(self) -> str:
        # We use "to_json_structure" here and not "to_reversible_json_structure"
        # because pickled values might not be deterministic.
        return utils.compute_digest(utils.to_json_structure(self._to_simple_structure(canonical=True)))

    def _to_simple_structure(self, *, canonical: bool = False) -> typing.Dict:
        problem_description = self

        if canonical:
            problem_description = self._canonical_problem_description(self)

        return dict(problem_description)

    def to_simple_structure(self, *, canonical: bool = False) -> typing.Dict:
        problem_description = self._to_simple_structure(canonical=canonical)

        problem_description['digest'] = self.get_digest()

        return problem_description

    @classmethod
    def from_simple_structure(cls: typing.Type[P], structure: typing.Dict, *, strict_digest: bool = False) -> P:
        return cls(structure, strict_digest=strict_digest)

    def to_json_structure(self, *, canonical: bool = False) -> typing.Dict:
        """
        For standard enumerations we map them to strings. Non-standard problem
        description fields we convert in a reversible manner.
        """

        PROBLEM_SCHEMA_VALIDATOR.validate(self)

        simple_structure = copy.deepcopy(self.to_simple_structure(canonical=canonical))

        if simple_structure.get('problem', {}).get('task_keywords', []):
            simple_structure['problem']['task_keywords'] = [task_keyword.name for task_keyword in simple_structure['problem']['task_keywords']]
        if simple_structure.get('problem', {}).get('performance_metrics', []):
            for metric in simple_structure['problem']['performance_metrics']:
                metric['metric'] = metric['metric'].name

        return utils.to_reversible_json_structure(simple_structure)

    @classmethod
    def from_json_structure(cls: typing.Type[P], structure: typing.Dict, *, strict_digest: bool = False) -> P:
        """
        For standard enumerations we map them from strings. For non-standard problem
        description fields we used a reversible conversion.
        """

        simple_structure = utils.from_reversible_json_structure(structure)

        # Legacy (before v4.0.0).
        legacy_task_keywords: typing.List[TaskKeyword] = []  # type: ignore
        legacy_task_keywords += JSON_TASK_TYPE_TO_KEYWORDS_MAP[simple_structure.get('problem', {}).get('task_type', None)]
        legacy_task_keywords += JSON_TASK_TYPE_TO_KEYWORDS_MAP[simple_structure.get('problem', {}).get('task_subtype', None)]

        if legacy_task_keywords:
            # We know "problem" field exists.
            simple_structure['problem']['task_keywords'] = simple_structure['problem'].get('task_keywords', []) + legacy_task_keywords

        if simple_structure.get('problem', {}).get('task_keywords', []):
            mapped_task_keywords = []
            for task_keyword in simple_structure['problem']['task_keywords']:
                if isinstance(task_keyword, str):
                    mapped_task_keywords.append(TaskKeyword[task_keyword])
                else:
                    mapped_task_keywords.append(task_keyword)
            simple_structure['problem']['task_keywords'] = mapped_task_keywords
        if simple_structure.get('problem', {}).get('performance_metrics', []):
            for metric in simple_structure['problem']['performance_metrics']:
                if isinstance(metric['metric'], str):
                    metric['metric'] = PerformanceMetric[metric['metric']]

        return cls.from_simple_structure(simple_structure, strict_digest=strict_digest)


@deprecate.function(message="use Problem.load class method instead")
def parse_problem_description(problem_doc_path: str) -> Problem:
    """
    Parses problem description according to ``problem.json`` metadata schema.

    It converts constants to enumerations when suitable.

    Parameters
    ----------
    problem_doc_path:
        File path to the problem description (``problemDoc.json``).

    Returns
    -------
    A parsed problem.
    """

    return Problem.load(problem_uri=utils.fix_uri(problem_doc_path))


def problem_serializer(obj: Problem) -> dict:
    data: typing.Dict = {
        'problem': dict(obj),
    }

    if type(obj) is not Problem:
        data['type'] = type(obj)

    return data


def problem_deserializer(data: dict) -> Problem:
    problem = data.get('type', Problem)(data['problem'])
    return problem


if pyarrow_lib is not None:
    pyarrow_lib._default_serialization_context.register_type(
        Problem, 'd3m.problem',
        custom_serializer=problem_serializer,
        custom_deserializer=problem_deserializer,
    )


def get_problem(problem_uri: str, *, strict_digest: bool = False, datasets_dir: str = None, handle_score_split: bool = True) -> Problem:
    if datasets_dir is not None:
        datasets, problem_descriptions = utils.get_datasets_and_problems(datasets_dir, handle_score_split)

        if problem_uri in problem_descriptions:
            problem_uri = problem_descriptions[problem_uri]

    problem_uri = utils.fix_uri(problem_uri)

    return Problem.load(problem_uri, strict_digest=strict_digest)


def describe_handler(
    arguments: argparse.Namespace, *, problem_resolver: typing.Callable = None,
) -> None:
    if problem_resolver is None:
        problem_resolver = get_problem

    output_stream = getattr(arguments, 'output', sys.stdout)

    has_errored = False

    for problem_path in arguments.problems:
        if getattr(arguments, 'list', False):
            print(problem_path, file=output_stream)

        try:
            problem = problem_resolver(problem_path, strict_digest=getattr(arguments, 'strict_digest', False))
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error parsing problem: {problem_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error parsing problem: {problem_path}") from error

        try:
            problem_description = problem.to_json_structure(canonical=True)

            if getattr(arguments, 'print', False):
                pprint.pprint(problem_description, stream=output_stream)
            elif not getattr(arguments, 'no_print', False):
                json.dump(
                    problem_description,
                    output_stream,
                    indent=(getattr(arguments, 'indent', 2) or None),
                    sort_keys=getattr(arguments, 'sort_keys', False),
                    allow_nan=False,
                )  # type: ignore
                output_stream.write('\n')
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error describing problem: {problem_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error describing problem: {problem_path}") from error

    if has_errored:
        sys.exit(1)


def main(argv: typing.Sequence) -> None:
    raise exceptions.NotSupportedError("This CLI has been removed. Use \"python3 -m d3m problem describe\" instead.")


if __name__ == '__main__':
    main(sys.argv)
