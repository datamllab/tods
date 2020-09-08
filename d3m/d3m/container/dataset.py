import abc
import argparse
import collections
import datetime
import errno
import filecmp
import hashlib
import io
import itertools
import json
import logging
import math
import os
import os.path
import pprint
import re
import shutil
import sys
import time
import traceback
import typing
from urllib import error as urllib_error, parse as url_parse

import dateutil.parser  # type: ignore
import frozendict  # type: ignore
import numpy  # type: ignore
import openml  # type: ignore
import pandas  # type: ignore
from pandas.io import common as pandas_io_common  # type: ignore
from sklearn import datasets  # type: ignore

from . import pandas as container_pandas
from d3m import deprecate, exceptions, utils
from d3m.metadata import base as metadata_base

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import lib as pyarrow_lib  # type: ignore
except ModuleNotFoundError:
    pyarrow_lib = None

__all__ = ('Dataset', 'ComputeDigest')

logger = logging.getLogger(__name__)

UNITS = {
    'B': 1, 'KB': 10**3, 'MB': 10**6, 'GB': 10**9, 'TB': 10**12, 'PB': 10**15,
    'KiB': 2*10, 'MiB': 2*20, 'GiB': 2*30, 'TiB': 2*40, 'PiB': 2*50,
}
SIZE_TO_UNITS = {
    1: 'B', 3: 'KB', 6: 'MB',
    9: 'GB', 12: 'TB', 15: 'PB',
}

D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES = {
    'index': 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
    'multiIndex': 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
    'key': 'https://metadata.datadrivendiscovery.org/types/UniqueKey',
    'attribute': 'https://metadata.datadrivendiscovery.org/types/Attribute',
    'suggestedTarget': 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
    'timeIndicator': 'https://metadata.datadrivendiscovery.org/types/Time',
    'locationIndicator': 'https://metadata.datadrivendiscovery.org/types/Location',
    'boundaryIndicator': 'https://metadata.datadrivendiscovery.org/types/Boundary',
    'interval': 'https://metadata.datadrivendiscovery.org/types/Interval',
    'instanceWeight': 'https://metadata.datadrivendiscovery.org/types/InstanceWeight',
    'boundingPolygon': 'https://metadata.datadrivendiscovery.org/types/BoundingPolygon',
    'suggestedPrivilegedData': 'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData',
    'suggestedGroupingKey': 'https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey',
    'edgeSource': 'https://metadata.datadrivendiscovery.org/types/EdgeSource',
    'directedEdgeSource': 'https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource',
    'undirectedEdgeSource': 'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource',
    'simpleEdgeSource': 'https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource',
    'multiEdgeSource': 'https://metadata.datadrivendiscovery.org/types/MultiEdgeSource',
    'edgeTarget': 'https://metadata.datadrivendiscovery.org/types/EdgeTarget',
    'directedEdgeTarget': 'https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget',
    'undirectedEdgeTarget': 'https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget',
    'simpleEdgeTarget': 'https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget',
    'multiEdgeTarget': 'https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget',
}

D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES = {
    # File collections.
    'image': 'http://schema.org/ImageObject',
    'video': 'http://schema.org/VideoObject',
    'audio': 'http://schema.org/AudioObject',
    'text': 'http://schema.org/Text',
    'speech': 'https://metadata.datadrivendiscovery.org/types/Speech',
    'timeseries': 'https://metadata.datadrivendiscovery.org/types/Timeseries',
    'raw': 'https://metadata.datadrivendiscovery.org/types/UnspecifiedStructure',
    # Other.
    'graph': 'https://metadata.datadrivendiscovery.org/types/Graph',
    'edgeList': 'https://metadata.datadrivendiscovery.org/types/EdgeList',
    'table': 'https://metadata.datadrivendiscovery.org/types/Table',
}

D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES = {
    'boolean': 'http://schema.org/Boolean',
    'integer': 'http://schema.org/Integer',
    'real': 'http://schema.org/Float',
    'string': 'http://schema.org/Text',
    'categorical': 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
    'dateTime': 'http://schema.org/DateTime',
    'realVector': 'https://metadata.datadrivendiscovery.org/types/FloatVector',
    'json': 'https://metadata.datadrivendiscovery.org/types/JSON',
    'geojson': 'https://metadata.datadrivendiscovery.org/types/GeoJSON',
    'unknown': 'https://metadata.datadrivendiscovery.org/types/UnknownType',
}

SEMANTIC_TYPES_TO_D3M_RESOURCE_TYPES = {v: k for k, v in D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES.items()}
SEMANTIC_TYPES_TO_D3M_ROLES = {v: k for k, v in D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES.items()}
SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES = {v: k for k, v in D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES.items()}

D3M_TO_DATASET_FIELDS: typing.Dict[typing.Sequence[str], typing.Tuple[typing.Sequence[str], bool]] = {
    ('about', 'datasetID'): (('id',), True),
    ('about', 'datasetName'): (('name',), True),
    ('about', 'description'): (('description',), False),
    ('about', 'datasetVersion'): (('version',), False),
    ('about', 'digest'): (('digest',), False),
    ('about', 'approximateSize'): (('approximate_stored_size',), False),
    ('about', 'citation'): (('source', 'citation'), False),
    ('about', 'license'): (('source', 'license'), False),
    ('about', 'redacted'): (('source', 'redacted'), False),
    ('about', 'source'): (('source', 'name'), False),
    ('about', 'citation'): (('source', 'citation'), False),
    ('about', 'humanSubjectsResearch'): (('source', 'human_subjects_research'), False),
}

INTERVAL_SEMANTIC_TYPES = (
    'https://metadata.datadrivendiscovery.org/types/IntervalStart',
    'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
)

BOUNDARY_SEMANTIC_TYPES = (
    'https://metadata.datadrivendiscovery.org/types/Interval',
    'https://metadata.datadrivendiscovery.org/types/BoundingPolygon',
) + INTERVAL_SEMANTIC_TYPES

# A map between legacy (before v4.0.0) D3M resource formats and media types.
# Now all resource formats are media types.
MEDIA_TYPES = {
    'audio/aiff': 'audio/aiff',
    'audio/flac': 'audio/flac',
    'audio/ogg': 'audio/ogg',
    'audio/wav': 'audio/wav',
    'audio/mpeg': 'audio/mpeg',
    'image/jpeg': 'image/jpeg',
    'image/png': 'image/png',
    'video/mp4': 'video/mp4',
    'video/avi': 'video/avi',
    'text/csv': 'text/csv',
    'text/csv+gzip': 'text/csv+gzip',
    'text/plain': 'text/plain',
    # Legacy (before v4.0.0) resource type for GML files.
    # In "MEDIA_TYPES_REVERSE" it is not present on purpose.
    'text/gml': 'text/vnd.gml',
    'text/vnd.gml': 'text/vnd.gml',
}
MEDIA_TYPES_REVERSE = {v: k for k, v in MEDIA_TYPES.items()}

# A legacy (before v4.0.0) map between D3M file extensions and media types.
# Now all datasets include a mapping between resource formats and file extensions.
# Based on: https://gitlab.com/datadrivendiscovery/data-supply/blob/shared/documentation/supportedResourceTypesFormats.json
FILE_EXTENSIONS = {
    '.aif': 'audio/aiff',
    '.aiff': 'audio/aiff',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.jpeg': 'image/jpeg',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
    '.csv': 'text/csv',
    '.csv.gz': 'text/csv+gzip',
    '.gml': 'text/vnd.gml',
    '.txt': 'text/plain',
    '.mp4': 'video/mp4',
    '.avi': 'video/avi',
}
FILE_EXTENSIONS_REVERSE: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)
for k, v in FILE_EXTENSIONS.items():
    FILE_EXTENSIONS_REVERSE[v].append(k)

TIME_GRANULARITIES = {
    'seconds': 'SECONDS',
    'minutes': 'MINUTES',
    'days': 'DAYS',
    'weeks': 'WEEKS',
    'months': 'MONTHS',
    'years': 'YEARS',
    'unspecified': 'UNSPECIFIED',
}
TIME_GRANULARITIES_REVERSE = {v: k for k, v in TIME_GRANULARITIES.items()}

ALL_D3M_SEMANTIC_TYPES = \
    set(D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES.values()) | \
    set(D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES.values()) | \
    set(D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES.values()) | \
    set(BOUNDARY_SEMANTIC_TYPES)

# A map between OpenML qualities and D3M metafeatures.
OPENML_QUALITY_MAP: typing.Dict[str, typing.Tuple[str, typing.Callable]] = {
    'Dimensionality': ('dimensionality', float),
    'NumberOfFeatures': ('number_of_attributes', int),
    'NumberOfInstances': ('number_of_instances', int),
    'NumberOfInstancesWithMissingValues': ('number_of_instances_with_missing_values', int),
    'PercentageOfInstancesWithMissingValues': ('ratio_of_instances_with_missing_values', float),
    'NumberOfMissingValues': ('number_of_missing_values', int),
    'PercentageOfMissingValues': ('ratio_of_missing_values', float),
    'NumberOfNumericFeatures': ('number_of_numeric_attributes', int),
    'PercentageOfNumericFeatures': ('ratio_of_numeric_attributes', float),
    'NumberOfBinaryFeatures': ('number_of_binary_attributes', int),
    'PercentageOfBinaryFeatures': ('ratio_of_binary_attributes', float),
    'NumberOfSymbolicFeatures': ('number_of_categorical_attributes', int),
    'PercentageOfSymbolicFeatures': ('ratio_of_categorical_attributes', float),
    'MeanNoiseToSignalRatio': ('noise_to_signal_ratio', float),
    'EquivalentNumberOfAtts': ('equivalent_number_of_attributes', int),
}

OPENML_IGNORED_QUALITIES = {
    # We use "number_distinct_values" on a target column instead.
    'NumberOfClasses',
    # We use "value_counts_aggregate.max" on a target column instead.
    'MajorityClassSize',
    # We use "value_probabilities_aggregate.max" on a target column instead.
    'MajorityClassPercentage',
    # We use "value_counts_aggregate.min" on a target column instead.
    'MinorityClassSize',
    # We use "value_probabilities_aggregate.min" on a target column instead.
    'MinorityClassPercentage',
    # We use "entropy_of_values" on a target column instead.
    'ClassEntropy',
    # It depends on the order of instances in the dataset, so it is a strange metafeature.
    # See: https://github.com/openml/EvaluationEngine/issues/34
    'AutoCorrelation',
    # The following are not computed by code availble through primitives, and we require that.
    'CfsSubsetEval_DecisionStumpAUC',
    'CfsSubsetEval_DecisionStumpErrRate',
    'CfsSubsetEval_DecisionStumpKappa',
    'CfsSubsetEval_NaiveBayesAUC',
    'CfsSubsetEval_NaiveBayesErrRate',
    'CfsSubsetEval_NaiveBayesKappa',
    'CfsSubsetEval_kNN1NAUC',
    'CfsSubsetEval_kNN1NErrRate',
    'CfsSubsetEval_kNN1NKappa',
    'DecisionStumpAUC',
    'DecisionStumpErrRate',
    'DecisionStumpKappa',
    'J48.00001.AUC',
    'J48.00001.ErrRate',
    'J48.00001.Kappa',
    'J48.0001.AUC',
    'J48.0001.ErrRate',
    'J48.0001.Kappa',
    'J48.001.AUC',
    'J48.001.ErrRate',
    'J48.001.Kappa',
    'REPTreeDepth1AUC',
    'REPTreeDepth1ErrRate',
    'REPTreeDepth1Kappa',
    'REPTreeDepth2AUC',
    'REPTreeDepth2ErrRate',
    'REPTreeDepth2Kappa',
    'REPTreeDepth3AUC',
    'REPTreeDepth3ErrRate',
    'REPTreeDepth3Kappa',
    'RandomTreeDepth1AUC',
    'RandomTreeDepth1ErrRate',
    'RandomTreeDepth1Kappa',
    'RandomTreeDepth2AUC',
    'RandomTreeDepth2ErrRate',
    'RandomTreeDepth2Kappa',
    'RandomTreeDepth3AUC',
    'RandomTreeDepth3ErrRate',
    'RandomTreeDepth3Kappa',
    'kNN1NAUC',
    'kNN1NErrRate',
    'kNN1NKappa',
    'NaiveBayesAUC',
    'NaiveBayesErrRate',
    'NaiveBayesKappa',
}

# A map between OpenML qualities and aggregated D3M metafeatures.
OPENML_QUALITY_AGGREGATE_MAP: typing.Dict[str, typing.Tuple[str, str, typing.Callable]] = {
    'MinAttributeEntropy': ('entropy_of_attributes', 'min', float),
    'MeanAttributeEntropy': ('entropy_of_attributes', 'mean', float),
    'MaxAttributeEntropy': ('entropy_of_attributes', 'max', float),
    'Quartile1AttributeEntropy': ('entropy_of_attributes', 'quartile_1', float),
    'Quartile2AttributeEntropy': ('entropy_of_attributes', 'median', float),
    'Quartile3AttributeEntropy': ('entropy_of_attributes', 'quartile_3', float),
    'MinSkewnessOfNumericAtts': ('skew_of_attributes', 'min', float),
    'MeanSkewnessOfNumericAtts': ('skew_of_attributes', 'mean', float),
    'MaxSkewnessOfNumericAtts': ('skew_of_attributes', 'max', float),
    'Quartile1SkewnessOfNumericAtts': ('skew_of_attributes', 'quartile_1', float),
    'Quartile2SkewnessOfNumericAtts': ('skew_of_attributes', 'median', float),
    'Quartile3SkewnessOfNumericAtts': ('skew_of_attributes', 'quartile_3', float),
    'MinMutualInformation': ('mutual_information_of_attributes', 'min', float),
    'MeanMutualInformation': ('mutual_information_of_attributes', 'mean', float),
    'MaxMutualInformation': ('mutual_information_of_attributes', 'max', float),
    'Quartile1MutualInformation': ('mutual_information_of_attributes', 'quartile_1', float),
    'Quartile2MutualInformation': ('mutual_information_of_attributes', 'median', float),
    'Quartile3MutualInformation': ('mutual_information_of_attributes', 'quartile_3', float),
    'MinMeansOfNumericAtts': ('mean_of_attributes', 'min', float),
    'MaxMeansOfNumericAtts': ('mean_of_attributes', 'max', float),
    'MeanMeansOfNumericAtts': ('mean_of_attributes', 'mean', float),
    'Quartile1MeansOfNumericAtts': ('mean_of_attributes', 'quartile_1', float),
    'Quartile2MeansOfNumericAtts': ('mean_of_attributes', 'median', float),
    'Quartile3MeansOfNumericAtts': ('mean_of_attributes', 'quartile_3', float),
    'MaxStdDevOfNumericAtts': ('standard_deviation_of_attributes', 'max', float),
    'MinStdDevOfNumericAtts': ('standard_deviation_of_attributes', 'min', float),
    'MeanStdDevOfNumericAtts': ('standard_deviation_of_attributes', 'mean', float),
    'Quartile1StdDevOfNumericAtts': ('standard_deviation_of_attributes', 'quartile_1', float),
    'Quartile2StdDevOfNumericAtts': ('standard_deviation_of_attributes', 'median', float),
    'Quartile3StdDevOfNumericAtts': ('standard_deviation_of_attributes', 'quartile_3', float),
    'MinNominalAttDistinctValues': ('number_distinct_values_of_categorical_attributes', 'min', float),
    'MaxNominalAttDistinctValues': ('number_distinct_values_of_categorical_attributes', 'max', float),
    'MeanNominalAttDistinctValues': ('number_distinct_values_of_categorical_attributes', 'mean', float),
    'StdvNominalAttDistinctValues': ('number_distinct_values_of_categorical_attributes', 'std', float),
    'MinKurtosisOfNumericAtts': ('kurtosis_of_attributes', 'min', float),
    'MaxKurtosisOfNumericAtts': ('kurtosis_of_attributes', 'max', float),
    'MeanKurtosisOfNumericAtts': ('kurtosis_of_attributes', 'mean', float),
    'Quartile1KurtosisOfNumericAtts': ('kurtosis_of_attributes', 'quartile_1', float),
    'Quartile2KurtosisOfNumericAtts': ('kurtosis_of_attributes', 'median', float),
    'Quartile3KurtosisOfNumericAtts': ('kurtosis_of_attributes', 'quartile_3', float),
}

OPENML_ID_REGEX = re.compile(r'^/d/(\d+)$')

DEFAULT_DATETIME = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

if not ALL_D3M_SEMANTIC_TYPES <= metadata_base.ALL_SEMANTIC_TYPES:
    raise ValueError("Not all D3M semantic types are defined in metadata.")


class ComputeDigest(utils.Enum):
    """
    Enumeration of possible approaches to computing dataset digest.
    """

    NEVER = 'NEVER'
    ONLY_IF_MISSING = 'ONLY_IF_MISSING'
    ALWAYS = 'ALWAYS'


def _add_extension_dot(extension: str) -> str:
    if not extension.startswith('.'):
        return '.' + extension
    return extension


def _remove_extension_dot(extension: str) -> str:
    if extension.startswith('.'):
        return extension[1:]
    return extension


def parse_size(size_string: str) -> int:
    number, unit = [string.strip() for string in size_string.split()]
    return int(float(number) * UNITS[unit])


def is_simple_boundary(semantic_types: typing.Tuple[str]) -> bool:
    """
    A simple boundary is a column with only "https://metadata.datadrivendiscovery.org/types/Boundary"
    semantic type and no other.
    """

    return 'https://metadata.datadrivendiscovery.org/types/Boundary' in semantic_types and not any(boundary_semantic_type in semantic_types for boundary_semantic_type in BOUNDARY_SEMANTIC_TYPES)


def update_digest(hash: typing.Any, file_path: str) -> None:
    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(hash.block_size)
            if not chunk:
                break
            hash.update(chunk)


# This exists as a reference implementation for computing a digest of D3M dataset.
# Loader below does an equivalent computation as part of dataset loading process.
def get_d3m_dataset_digest(dataset_doc_path: str) -> str:
    hash = hashlib.sha256()

    with open(dataset_doc_path, 'r', encoding='utf8') as dataset_doc_file:
        dataset_doc = json.load(dataset_doc_file)

    dataset_path = os.path.dirname(dataset_doc_path)

    for data_resource in dataset_doc['dataResources']:
        if data_resource.get('isCollection', False):
            collection_path = os.path.join(dataset_path, data_resource['resPath'])

            # We assume that we can just concat "collection_path" with a value in the column.
            assert collection_path[-1] == '/'

            for filename in utils.list_files(collection_path):
                file_path = os.path.join(collection_path, filename)

                # We include both the filename and the content.
                hash.update(os.path.join(data_resource['resPath'], filename).encode('utf8'))
                update_digest(hash, file_path)

        else:
            resource_path = os.path.join(dataset_path, data_resource['resPath'])

            # We include both the filename and the content.
            hash.update(data_resource['resPath'].encode('utf8'))
            update_digest(hash, resource_path)

    # We remove digest, if it exists in dataset description, before computing the digest over the rest.
    dataset_doc['about'].pop('digest', None)

    # We add to hash also the dataset description, with sorted keys.
    hash.update(json.dumps(dataset_doc, sort_keys=True).encode('utf8'))

    return hash.hexdigest()


class Loader(metaclass=utils.AbstractMetaclass):
    """
    A base class for dataset loaders.
    """

    @abc.abstractmethod
    def can_load(self, dataset_uri: str) -> bool:
        """
        Return ``True`` if this loader can load a dataset from a given URI ``dataset_uri``.

        Parameters
        ----------
        dataset_uri:
            A URI to load a dataset from.

        Returns
        -------
        ``True`` if this loader can load a dataset from ``dataset_uri``.
        """

    @abc.abstractmethod
    def load(self, dataset_uri: str, *, dataset_id: str = None, dataset_version: str = None, dataset_name: str = None, lazy: bool = False,
             compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False, handle_score_split: bool = True) -> 'Dataset':
        """
        Loads the dataset at ``dataset_uri``.

        Parameters
        ----------
        dataset_uri:
            A URI to load.
        dataset_id:
            Override dataset ID determined by the loader.
        dataset_version:
            Override dataset version determined by the loader.
        dataset_name:
            Override dataset name determined by the loader.
        lazy:
            If ``True``, load only top-level metadata and not whole dataset.
        compute_digest:
            Compute a digest over the data?
        strict_digest:
            If computed digest does not match the one provided in metadata, raise an exception?
        handle_score_split:
            If a scoring dataset has target values in a separate file, merge them in?

        Returns
        -------
        A loaded dataset.
        """


class Saver(metaclass=utils.AbstractMetaclass):
    """
    A base class for dataset savers.
    """

    @abc.abstractmethod
    def can_save(self, dataset_uri: str) -> bool:
        """
        Return ``True`` if this saver can save a dataset to a given URI ``dataset_uri``.

        Parameters
        ----------
        dataset_uri:
            A URI to save a dataset to.

        Returns
        -------
        ``True`` if this saver can save a dataset to ``dataset_uri``.
        """

    @abc.abstractmethod
    def save(self, dataset: 'Dataset', dataset_uri: str, *, compute_digest: ComputeDigest = ComputeDigest.ALWAYS, preserve_metadata: bool = True) -> None:
        """
        Saves the dataset ``dataset`` to ``dataset_uri``.

        Parameters
        ----------
        dataset:
            A dataset to save.
        dataset_uri:
            A URI to save to.
        compute_digest:
            Compute digest over the data when saving?
        preserve_metadata:
            When saving a dataset, store its metadata as well?
        """


class OpenMLDatasetLoader(Loader):
    """
    A class for loading OpenML datasets.
    """

    def can_load(self, dataset_uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(dataset_uri)
        except Exception:
            return False

        if parsed_uri.scheme != 'https':
            return False

        if 'www.openml.org' != parsed_uri.netloc:
            return False

        if OPENML_ID_REGEX.search(parsed_uri.path) is None:
            return False

        return True

    def _load_data(self, openml_dataset: openml.OpenMLDataset, resources: typing.Dict, metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        # OpenML package always computes digests when downloading data and checks them, failing if they do not match.
        # See: https://github.com/openml/OpenML/issues/1027
        data, _, categorical_indicator, column_names = openml_dataset.get_data(include_row_id=True, include_ignore_attribute=True, dataset_format='dataframe')

        assert data.shape[1] == len(categorical_indicator)
        assert data.shape[1] == len(column_names)
        assert data.shape[1] == len(openml_dataset.features)
        assert set(data.columns) == set(column_names)

        if openml_dataset.ignore_attribute:
            if isinstance(openml_dataset.ignore_attribute, str):
                ignore_columns = set(openml_dataset.ignore_attribute.split(','))
            else:
                ignore_columns = set(openml_dataset.ignore_attribute)
        else:
            ignore_columns = set()

        assert ignore_columns <= set(column_names)

        if openml_dataset.default_target_attribute:
            if isinstance(openml_dataset.default_target_attribute, str):
                target_columns = set(openml_dataset.default_target_attribute.split(','))
            else:
                target_columns = set(openml_dataset.default_target_attribute)
        else:
            target_columns = set()

        assert target_columns <= set(column_names)

        openml_column_data_types = {}
        for i, column_name in enumerate(column_names):
            openml_column_data_types[column_name] = openml_dataset.features[i].data_type

            assert (openml_column_data_types[column_name] == 'nominal' and categorical_indicator[i]) or (openml_column_data_types[column_name] != 'nominal' and not categorical_indicator[i])

            # For nominal data types we store a list of possible values.
            if openml_column_data_types[column_name] == 'nominal':
                openml_column_data_types[column_name] = openml_dataset.features[i].nominal_values

        data = self._convert_categorical_columns(data, categorical_indicator)

        if openml_dataset.row_id_attribute:
            assert openml_dataset.row_id_attribute in column_names

            row_id_column = openml_dataset.row_id_attribute
        else:
            assert 'd3mIndex' not in column_names

            # We do not update digest with new data generated here. This is OK because this data is determined by
            # original data so original digest still applies. When saving a new digest has to be computed anyway
            # because this data will have to be converted to string.
            data.insert(0, 'd3mIndex', range(len(data)))

            column_names.insert(0, 'd3mIndex')
            categorical_indicator = [False] + list(categorical_indicator)
            openml_column_data_types['d3mIndex'] = 'integer'
            row_id_column = 'd3mIndex'

        data = container_pandas.DataFrame(data)

        resources['learningData'] = data
        metadata = metadata.update((), {
            'dimension': {'length': len(resources)},
        })

        metadata = metadata.update(('learningData',), {
            'structural_type': type(data),
            'dimension': {
                'length': len(data)
            },
        })
        metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'length': len(column_names)
            },
        })

        for column_index, column_name in enumerate(column_names):
            column_metadata = {
                'semantic_types': [
                    self._semantic_type(openml_column_data_types[column_name]),
                ],
                'name': column_name,
            }

            if column_name in target_columns:
                column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

            if column_name == row_id_column:
                column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
            elif column_name not in ignore_columns:
                column_metadata['semantic_types'].append('https://metadata.datadrivendiscovery.org/types/Attribute')

            if utils.is_sequence(openml_column_data_types[column_name]):
                # We convert all categorical columns into string columns.
                column_metadata['structural_type'] = str
            elif openml_column_data_types[column_name] == 'nominal':
                raise exceptions.InvalidStateError("Nominal column data type which has not been converted to a list of values.")
            elif openml_column_data_types[column_name] in ['string', 'date']:
                column_metadata['structural_type'] = str
            elif openml_column_data_types[column_name] == 'integer':
                column_metadata['structural_type'] = int
            else:
                column_metadata['structural_type'] = float

            metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, column_index), column_metadata)

        metadata = metadata.set_table_metadata(at=('learningData',))

        # Adding it here so that the order of semantic types is consistent between saving and loading of datasets.
        metadata = metadata.add_semantic_type(('learningData',), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return metadata

    def _get_dataset_metafeatures(self, openml_dataset: openml.OpenMLDataset) -> typing.Dict:
        openml_qualities = openml_dataset.qualities or {}
        metafeatures: typing.Dict = {}

        unknown_qualities = set(openml_qualities.keys()) - set(OPENML_QUALITY_MAP.keys()) - set(OPENML_QUALITY_AGGREGATE_MAP.keys()) - OPENML_IGNORED_QUALITIES
        if unknown_qualities:
            logger.warning("Unknown OpenML qualities in dataset %(dataset_id)s: %(unknown_qualities)s", {
                'dataset_id': openml_dataset.dataset_id,
                'unknown_qualities': sorted(unknown_qualities),
            })

        for quality_key, quality_value in openml_qualities.items():
            if numpy.isnan(quality_value):
                continue

            if quality_key in OPENML_IGNORED_QUALITIES:
                continue

            if quality_key in OPENML_QUALITY_MAP:
                mapped_quality, quality_type = OPENML_QUALITY_MAP[quality_key]

                metafeatures[mapped_quality] = quality_type(quality_value)

            elif quality_key in OPENML_QUALITY_AGGREGATE_MAP:
                mapped_quality, aggregate_key, quality_type = OPENML_QUALITY_AGGREGATE_MAP[quality_key]

                if mapped_quality not in metafeatures:
                    metafeatures[mapped_quality] = {}

                metafeatures[mapped_quality][aggregate_key] = quality_type(quality_value)

            # We warn about unknown qualities above.

        return metafeatures

    def _semantic_type(self, data_type: str) -> str:
        if utils.is_sequence(data_type):
            if len(data_type) == 2:
                return 'http://schema.org/Boolean'
            else:
                return 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
        elif data_type == 'integer':
            return 'http://schema.org/Integer'
        elif data_type == 'real':
            return 'http://schema.org/Float'
        elif data_type == 'numeric':
            return 'http://schema.org/Float'
        elif data_type == 'string':
            return 'http://schema.org/Text'
        elif data_type == 'date':
            return 'http://schema.org/DateTime'
        else:
            raise exceptions.UnexpectedValueError("Data type '{data_type}' is not supported.".format(data_type=data_type))

    def _get_dataset_metadata(self, openml_dataset: openml.OpenMLDataset) -> typing.Dict:
        """
        Returns OpenML only metadata converted to D3M metadata. It also computes digest using this metadata and expected data digest.
        """

        dataset_metadata: typing.Dict[str, typing.Any] = {
            'id': str(openml_dataset.dataset_id),
        }

        if openml_dataset.name:
            dataset_metadata['name'] = openml_dataset.name
        if openml_dataset.description:
            dataset_metadata['description'] = openml_dataset.description
        if openml_dataset.version_label:
            dataset_metadata['version'] = openml_dataset.version_label
        if openml_dataset.tag:
            dataset_metadata['keywords'] = openml_dataset.tag

        dataset_source: typing.Dict[str, typing.Any] = {
            'uris': []
        }

        if openml_dataset.creator:
            dataset_source['name'] = openml_dataset.creator
        if openml_dataset.licence:
            dataset_source['license'] = openml_dataset.licence
        if openml_dataset.citation:
            dataset_source['citation'] = openml_dataset.citation
        if openml_dataset.collection_date:
            dataset_source['published'] = utils.datetime_for_json(dateutil.parser.parse(openml_dataset.collection_date, default=DEFAULT_DATETIME, fuzzy=True))
        if openml_dataset.openml_url or openml_dataset.url:
            dataset_source['uris'].append(openml_dataset.openml_url or openml_dataset.url)
        if openml_dataset.original_data_url:
            dataset_source['uris'].append(openml_dataset.original_data_url)
        if openml_dataset.paper_url:
            dataset_source['uris'].append(openml_dataset.paper_url)

        if not dataset_source['uris']:
            del dataset_source['uris']
        if dataset_source:
            dataset_metadata['source'] = dataset_source

        if not openml_dataset.md5_checksum:
            raise exceptions.UnexpectedValueError("OpenML dataset {id} does not have MD5 checksum.".format(id=openml_dataset.dataset_id))

        dataset_metadata['digest'] = utils.compute_digest(dataset_metadata, openml_dataset.md5_checksum.encode('utf8'))

        return dataset_metadata

    def _convert_categorical_columns(self, data: pandas.DataFrame, categorical_indicator: typing.List[bool]) -> pandas.DataFrame:
        """
        Converts categorical DataFrame columns to str columns. In D3M pipelines generally expect categorical
        columns to be encoded as strings and only later the pipeline encodes them in some way.
        """

        for column_index, is_categorical in enumerate(categorical_indicator):
            if not is_categorical:
                continue

            column_name = data.columns[column_index]

            data[column_name] = data[column_name].astype(str)

        return data

    # "strict_digest" and "compute_digest" are ignored because OpenML package always computes digests when downloading data
    # and checks them, failing if they do not match. See: https://github.com/openml/OpenML/issues/1027
    # "handle_score_split" is ignored.
    def load(self, dataset_uri: str, *, dataset_id: str = None, dataset_version: str = None, dataset_name: str = None, lazy: bool = False,
             compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False, handle_score_split: bool = True) -> 'Dataset':
        assert self.can_load(dataset_uri)

        parsed_uri = url_parse.urlparse(dataset_uri, allow_fragments=False)
        dataset_path_id = OPENML_ID_REGEX.search(parsed_uri.path)[1]

        try:
            # We download just metadata first.
            openml_dataset = openml.datasets.get_dataset(dataset_path_id, download_data=False)
        except openml.exceptions.OpenMLServerException as error:
            raise exceptions.DatasetNotFoundError(
                "OpenML dataset '{dataset_uri}' cannot be found.".format(dataset_uri=dataset_uri),
            ) from error

        # This converts OpenML dataset metadata to D3M dataset metadata.
        dataset_metadata = self._get_dataset_metadata(openml_dataset)

        assert dataset_metadata['id'] == dataset_path_id

        # Use overrides if provided. Digest is not computed over those changes on purpose.
        if dataset_id is not None:
            dataset_metadata['id'] = dataset_id
        if dataset_version is not None:
            dataset_metadata['version'] = dataset_version
        if dataset_name is not None:
            dataset_metadata['name'] = dataset_name

        # Other standard metadata.
        dataset_metadata.update({
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': Dataset,
            'location_uris': [
                dataset_uri,
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                'length': 0,
            },
        })

        dataset_metafeatures = self._get_dataset_metafeatures(openml_dataset)
        if dataset_metafeatures:
            # We set metafeatures on the top level even if otherwise in D3M we set metafeatures at the resource level or
            # even target column level, but setting them here allows one to access them in the lazy mode (when there are
            # no resources yet in the dataset). We also do not include them into a digest because for D3M datasets
            # the digest is just about the stored files of the dataset and not any additional metadata added by the loader.
            dataset_metadata['data_metafeatures'] = dataset_metafeatures

        resources: typing.Dict = {}
        metadata = metadata_base.DataMetadata(dataset_metadata)

        if not lazy:
            load_lazy = None

            metadata = self._load_data(
                openml_dataset, resources, metadata,
            )

        else:
            def load_lazy(dataset: Dataset) -> None:
                # "dataset" can be used as "resources", it is a dict of values.
                dataset.metadata = self._load_data(
                    openml_dataset, dataset, dataset.metadata,
                )

                dataset._load_lazy = None

        return Dataset(resources, metadata, load_lazy=load_lazy)


class D3MDatasetLoader(Loader):
    """
    A class for loading of D3M datasets.

    Loader support only loading from a local file system.
    URI should point to the ``datasetDoc.json`` file in the D3M dataset directory.
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

        if os.path.basename(parsed_uri.path) != 'datasetDoc.json':
            return False

        return True

    def _load_data(self, resources: typing.Dict, metadata: metadata_base.DataMetadata, *, dataset_path: str, dataset_doc: typing.Dict,
                   dataset_id: typing.Optional[str], dataset_digest: typing.Optional[str],
                   compute_digest: ComputeDigest, strict_digest: bool, handle_score_split: bool) -> typing.Tuple[metadata_base.DataMetadata, typing.Optional[str]]:
        # Allowing "True" for backwards compatibility.
        if compute_digest is True or compute_digest == ComputeDigest.ALWAYS or (compute_digest == ComputeDigest.ONLY_IF_MISSING and dataset_digest is None):
            hash = hashlib.sha256()
        else:
            hash = None

        for data_resource in dataset_doc['dataResources']:
            if data_resource.get('isCollection', False):
                resources[data_resource['resID']], metadata = self._load_collection(dataset_path, data_resource, metadata, hash)
            else:
                loader = getattr(self, '_load_resource_type_{resource_type}'.format(resource_type=data_resource['resType']), None)
                if loader is None:
                    raise exceptions.NotSupportedError("Resource type '{resource_type}' is not supported.".format(resource_type=data_resource['resType']))

                resources[data_resource['resID']], metadata = loader(dataset_path, data_resource, metadata, hash)

        # Backwards compatibility. If there is no resource marked as a dataset entry point,
        # check if there is any resource with a suitable filename.
        for data_resource in dataset_doc['dataResources']:
            if metadata.has_semantic_type((data_resource['resID'],), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'):
                break
        else:
            for data_resource in dataset_doc['dataResources']:
                if os.path.splitext(os.path.basename(data_resource['resPath']))[0] == 'learningData':
                    metadata = metadata.add_semantic_type((data_resource['resID'],), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
        # They are the same as TEST dataset splits, but we present them differently, so that
        # SCORE dataset splits have targets as part of data.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
        if handle_score_split and os.path.exists(os.path.join(dataset_path, '..', 'targets.csv')):
            self._merge_score_targets(resources, metadata, dataset_path, hash)

        if hash is not None:
            # We remove digest, if it exists in dataset description, before computing the digest over the rest.
            # We modify "dataset_doc" here, but this is OK, we do not need it there anymore at this point.
            dataset_doc['about'].pop('digest', None)

            # We add to hash also the dataset description, with sorted keys.
            hash.update(json.dumps(dataset_doc, sort_keys=True).encode('utf8'))

            new_dataset_digest = hash.hexdigest()

            if dataset_digest is not None and dataset_digest != new_dataset_digest:
                if strict_digest:
                    raise exceptions.DigestMismatchError(
                        "Digest for dataset '{dataset_id}' does not match one from dataset description. Dataset description digest: {dataset_digest}. Computed digest: {new_dataset_digest}.".format(
                            dataset_id=dataset_id or dataset_doc['about']['datasetID'],
                            dataset_digest=dataset_digest,
                            new_dataset_digest=new_dataset_digest,
                        )
                    )
                else:
                    logger.warning(
                        "Digest for dataset '%(dataset_id)s' does not match one from dataset description. Dataset description digest: %(dataset_digest)s. Computed digest: %(new_dataset_digest)s.",
                        {
                            'dataset_id': dataset_id or dataset_doc['about']['datasetID'],
                            'dataset_digest': dataset_digest,
                            'new_dataset_digest': new_dataset_digest,
                        },
                    )
        else:
            new_dataset_digest = dataset_doc['about'].get('digest', None)

        return metadata, new_dataset_digest

    def load(self, dataset_uri: str, *, dataset_id: str = None, dataset_version: str = None, dataset_name: str = None, lazy: bool = False,
             compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False, handle_score_split: bool = True) -> 'Dataset':
        assert self.can_load(dataset_uri)

        parsed_uri = url_parse.urlparse(dataset_uri, allow_fragments=False)

        dataset_doc_path = parsed_uri.path
        dataset_path = os.path.dirname(dataset_doc_path)

        try:
            with open(dataset_doc_path, 'r', encoding='utf8') as dataset_doc_file:
                dataset_doc = json.load(dataset_doc_file)
        except FileNotFoundError as error:
            raise exceptions.DatasetNotFoundError(
                "D3M dataset '{dataset_uri}' cannot be found.".format(dataset_uri=dataset_uri),
            ) from error

        dataset_schema_version = dataset_doc.get('about', {}).get('datasetSchemaVersion', '3.3.0')
        if dataset_schema_version not in self.SUPPORTED_VERSIONS:
            logger.warning("Loading a dataset with unsupported schema version '%(version)s'. Supported versions: %(supported_versions)s", {
                'version': dataset_schema_version,
                'supported_versions': self.SUPPORTED_VERSIONS,
            })

        # We do not compute digest here, but we use one from dataset description if it exist.
        # This is different from other loaders which compute digest when lazy loading and check
        # it after data is finally loaded to make sure data has not changed in meantime.
        dataset_digest = dataset_doc['about'].get('digest', None)

        resources: typing.Dict = {}
        metadata = metadata_base.DataMetadata()

        metadata = self._load_top_qualities(dataset_doc, metadata)

        if not lazy:
            load_lazy = None

            metadata = self._load_data_qualities(dataset_doc, metadata)

            metadata, dataset_digest = self._load_data(
                resources, metadata, dataset_path=dataset_path, dataset_doc=dataset_doc, dataset_id=dataset_id,
                dataset_digest=dataset_digest, compute_digest=compute_digest, strict_digest=strict_digest,
                handle_score_split=handle_score_split,
            )

        else:
            def load_lazy(dataset: Dataset) -> None:
                nonlocal dataset_digest

                dataset.metadata = self._load_data_qualities(dataset_doc, dataset.metadata)

                # "dataset" can be used as "resources", it is a dict of values.
                dataset.metadata, dataset_digest = self._load_data(
                    dataset, dataset.metadata, dataset_path=dataset_path, dataset_doc=dataset_doc, dataset_id=dataset_id,
                    dataset_digest=dataset_digest, compute_digest=compute_digest, strict_digest=strict_digest,
                    handle_score_split=handle_score_split,
                )

                new_metadata = {
                    'dimension': {'length': len(dataset)},
                }

                if dataset_digest is not None:
                    new_metadata['digest'] = dataset_digest

                dataset.metadata = dataset.metadata.update((), new_metadata)
                dataset.metadata = dataset.metadata.generate(dataset)

                dataset._load_lazy = None

        document_dataset_id = dataset_doc['about']['datasetID']
        # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
        # They are the same as TEST dataset splits, but we present them differently, so that
        # SCORE dataset splits have targets as part of data. Because of this we also update
        # corresponding dataset ID.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
        if handle_score_split and os.path.exists(os.path.join(dataset_path, '..', 'targets.csv')) and document_dataset_id.endswith('_TEST'):
            document_dataset_id = document_dataset_id[:-5] + '_SCORE'

        dataset_metadata = {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': Dataset,
            'id': dataset_id or document_dataset_id,
            'name': dataset_name or dataset_doc['about']['datasetName'],
            'dimension': {
                'name': 'resources',
                'length': len(resources),
            },
        }

        if dataset_version or dataset_doc['about'].get('datasetVersion', None):
            dataset_metadata['version'] = dataset_version or dataset_doc['about']['datasetVersion']

        if dataset_digest is not None:
            dataset_metadata['digest'] = dataset_digest

        if dataset_doc['about'].get('description', None):
            dataset_metadata['description'] = dataset_doc['about']['description']

        if dataset_doc['about'].get('approximateSize', None):
            try:
                dataset_metadata['approximate_stored_size'] = parse_size(dataset_doc['about']['approximateSize'])
            except Exception as error:
                raise ValueError("Unable to parse 'approximateSize': {approximate_size}".format(approximate_size=dataset_doc['about']['approximateSize'])) from error

        dataset_source = {}

        if 'redacted' in dataset_doc['about']:
            dataset_source['redacted'] = dataset_doc['about']['redacted']

        # "license" is often an empty string and in that case we do not want
        # really to set the field in dataset metadata.
        if dataset_doc['about'].get('license', None):
            dataset_source['license'] = dataset_doc['about']['license']

        if 'humanSubjectsResearch' in dataset_doc['about']:
            dataset_source['human_subjects_research'] = dataset_doc['about']['humanSubjectsResearch']

        if dataset_doc['about'].get('source', None):
            dataset_source['name'] = dataset_doc['about']['source']

        if dataset_doc['about'].get('citation', None):
            dataset_source['citation'] = dataset_doc['about']['citation']

        if dataset_doc['about'].get('publicationDate', None):
            try:
                dataset_source['published'] = utils.datetime_for_json(dateutil.parser.parse(dataset_doc['about']['publicationDate'], default=DEFAULT_DATETIME, fuzzy=True))
            except Exception as error:
                raise ValueError("Unable to parse 'publicationDate': {publication_date}".format(publication_date=dataset_doc['about']['publicationDate'])) from error

        if dataset_source:
            dataset_metadata['source'] = dataset_source

        metadata = metadata.update((), dataset_metadata)

        # We reconstruct the URI to normalize it.
        location_uri = utils.fix_uri(dataset_doc_path)
        location_uris = list(metadata.query(()).get('location_uris', []))
        if location_uri not in location_uris:
            location_uris.insert(0, location_uri)
            metadata = metadata.update((), {'location_uris': location_uris})

        if dataset_doc['about'].get('datasetURI', None) and dataset_doc['about']['datasetURI'] not in location_uris:
            location_uris.append(dataset_doc['about']['datasetURI'])
            metadata = metadata.update((), {'location_uris': location_uris})

        semantic_types = list(metadata.query(()).get('dimension', {}).get('semantic_types', []))
        if 'https://metadata.datadrivendiscovery.org/types/DatasetResource' not in semantic_types:
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/DatasetResource')
            metadata = metadata.update((), {'dimension': {'semantic_types': semantic_types}})

        source_uris = list(metadata.query(()).get('source', {}).get('uris', []))
        if dataset_doc['about'].get('sourceURI', None) and dataset_doc['about']['sourceURI'] not in source_uris:
            source_uris.insert(0, dataset_doc['about']['sourceURI'])
            metadata = metadata.update((), {'source': {'uris': source_uris}})

        keywords = list(metadata.query(()).get('keywords', []))
        if dataset_doc['about'].get('applicationDomain', None) and dataset_doc['about']['applicationDomain'] not in keywords:
            # Application domain has no vocabulary specified so we map it to keywords.
            keywords.append(dataset_doc['about']['applicationDomain'])
            metadata.update((), {'keywords': keywords})

        return Dataset(resources, metadata, load_lazy=load_lazy)

    def _load_top_qualities(self, dataset_doc: typing.Dict, metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        ALL_ELEMENTS_REPR = repr(metadata_base.ALL_ELEMENTS)

        for quality in dataset_doc.get('qualities', []):
            restricted_to = quality.get('restrictedTo', {})

            # D3M metadata stored as D3M qualities.
            if quality['qualName'] == 'metadata':
                if restricted_to['resID'] == '':
                    selector: metadata_base.TupleSelector = ()
                else:
                    # Here we load only top-level metadata.
                    continue

                # TODO: Optimize, see: https://gitlab.com/datadrivendiscovery/d3m/issues/408
                metadata = metadata.update(selector, utils.from_reversible_json_structure(quality['qualValue']))

        return metadata

    def _load_data_qualities(self, dataset_doc: typing.Dict, metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        ALL_ELEMENTS_REPR = repr(metadata_base.ALL_ELEMENTS)

        for quality in dataset_doc.get('qualities', []):
            restricted_to = quality.get('restrictedTo', {})

            # D3M metadata stored as D3M qualities.
            if quality['qualName'] == 'metadata':
                if restricted_to['resID'] == '':
                    # Here we load only non top-level metadata.
                    continue
                else:
                    resource_selector = [metadata_base.ALL_ELEMENTS if segment == ALL_ELEMENTS_REPR else segment for segment in restricted_to['resComponent']['selector']]
                    selector: metadata_base.TupleSelector = (restricted_to['resID'], *resource_selector)

                # TODO: Optimize, see: https://gitlab.com/datadrivendiscovery/d3m/issues/408
                metadata = metadata.update(selector, utils.from_reversible_json_structure(quality['qualValue']))

            # An alternative way to describe LUPI datasets using D3M qualities.
            # See: https://gitlab.com/datadrivendiscovery/d3m/issues/61
            #      https://gitlab.com/datadrivendiscovery/d3m/issues/225
            elif quality['qualName'] == 'privilegedFeature':
                if quality['qualValue'] != 'True':
                    continue

                column_index = restricted_to.get('resComponent', {}).get('columnIndex', None)
                if column_index is not None:
                    metadata = self._add_semantic_type_for_column_index(metadata, restricted_to['resID'], column_index, 'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData')
                    continue

                column_name = restricted_to.get('resComponent', {}).get('columnName', None)
                if column_name is not None:
                    metadata = self._add_semantic_type_for_column_name(metadata, restricted_to['resID'], column_name, 'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData')
                    continue

        return metadata

    def _add_semantic_type_for_column_index(self, metadata: metadata_base.DataMetadata, resource_id: str, column_index: int, semantic_type: str) -> metadata_base.DataMetadata:
        return metadata.add_semantic_type((resource_id, metadata_base.ALL_ELEMENTS, column_index), semantic_type)

    def _add_semantic_type_for_column_name(self, metadata: metadata_base.DataMetadata, resource_id: str, column_name: str, semantic_type: str) -> metadata_base.DataMetadata:
        column_index = metadata.get_column_index_from_column_name(column_name, at=(resource_id,))

        return self._add_semantic_type_for_column_index(metadata, resource_id, column_index, semantic_type)

    def _load_collection(self, dataset_path: str, data_resource: typing.Dict, metadata: metadata_base.DataMetadata,
                         hash: typing.Any) -> typing.Tuple[container_pandas.DataFrame, metadata_base.DataMetadata]:
        assert data_resource.get('isCollection', False)

        collection_path = os.path.join(dataset_path, data_resource['resPath'])

        media_types_with_extensions = {}
        # Legacy (before v4.0.0). We obtain a list of file extensions from the global list of file extensions.
        if utils.is_sequence(data_resource['resFormat']):
            for format in data_resource['resFormat']:
                format_media_type = MEDIA_TYPES[format]
                media_types_with_extensions[format_media_type] = [_add_extension_dot(extension) for extension in FILE_EXTENSIONS_REVERSE[format_media_type]]
        else:
            for format, extensions in data_resource['resFormat'].items():
                # We allow unknown formats, hoping that they are proper media types already.
                format_media_type = MEDIA_TYPES.get(format, format)
                # We do not really care if file extensions are not on the global list of file extensions.
                media_types_with_extensions[format_media_type] = [_add_extension_dot(extension) for extension in extensions]

        all_media_types_set = set(media_types_with_extensions.keys())

        reverse_media_types_with_extensions: typing.Dict[str, str] = {}
        for media_type, extensions in media_types_with_extensions.items():
            for extension in extensions:
                if extension in reverse_media_types_with_extensions:
                    raise exceptions.InvalidDatasetError("Conflicting file extension '{file_extension}': {media_type1} and {media_type2}".format(
                        file_extension=extension,
                        media_type1=reverse_media_types_with_extensions[extension],
                        media_type2=media_type,
                    ))

                reverse_media_types_with_extensions[extension] = media_type

        filenames = []
        media_types = []

        for filename in utils.list_files(collection_path):
            file_path = os.path.join(collection_path, filename)

            filename_extension = os.path.splitext(filename)[1]

            filenames.append(filename)

            try:
                media_type = reverse_media_types_with_extensions[filename_extension]
            except KeyError as error:
                raise TypeError("Unable to determine a media type for the file extension of file '{filename}'.".format(filename=filename)) from error

            media_types.append(media_type)

            if hash is not None:
                # We include both the filename and the content.
                hash.update(os.path.join(data_resource['resPath'], filename).encode('utf8'))
                update_digest(hash, file_path)

        data = container_pandas.DataFrame({'filename': filenames}, columns=['filename'], dtype=object)

        metadata = metadata.update((data_resource['resID'],), {
            'structural_type': type(data),
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/FilesCollection',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': len(data),
            },
        })

        metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 1,
            },
        })

        location_base_uri = utils.fix_uri(collection_path)
        # We want to make sure you can just concat with the filename.
        if not location_base_uri.endswith('/'):
            location_base_uri += '/'

        media_types_set = set(media_types)

        extra_media_types = all_media_types_set - media_types_set
        if extra_media_types:
            logger.warning("File collection '%(resource_id)s' claims more file formats than are used in files. Extraneous formats: %(formats)s", {
                'resource_id': data_resource['resID'],
                'formats': [MEDIA_TYPES_REVERSE.get(format, format) for format in sorted(extra_media_types)],
            })

        # Normalize the list based on real media types used.
        all_media_types = sorted(media_types_set)

        column_metadata = {
            'name': 'filename',
            'structural_type': str,
            'location_base_uris': [
                location_base_uri,
            ],
            # A superset of all media types of files in this collection.
            'media_types': all_media_types,
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                'https://metadata.datadrivendiscovery.org/types/FileName',
                D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[data_resource['resType']],
            ],
        }

        if data_resource.get('columns', None):
            columns_metadata = []

            for column in data_resource['columns']:
                columns_metadata.append(self._get_column_metadata(column))
                columns_metadata[-1]['column_index'] = column['colIndex']
                columns_metadata[-1]['column_name'] = column['colName']

            column_metadata['file_columns'] = columns_metadata

        if data_resource.get('columnsCount', None) is not None:
            column_metadata['file_columns_count'] = data_resource['columnsCount']

        metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS, 0), column_metadata)

        # If there are different rows with different media types, we have to set
        # on each row which media type it is being used.
        if len(all_media_types) > 1:
            # The following modifies metadata for rows directly instead of through metadata methods
            # to achieve useful performance because some datasets contain many files which means many
            # rows have their "media_types" set. Setting it one by one makes things to slow.
            # Here we are taking advantage of quite few assumptions: we are modifying metadata in-place
            # because we know it is only us having a reference to it, we directly set metadata for
            # rows because we know no other metadata exists for rows, moreover, we also know no other
            # metadata exists for rows through any higher ALL_ELEMENTS.
            # TODO: Expose this as a general metadata method.
            # TODO: Or just optimize, see: https://gitlab.com/datadrivendiscovery/d3m/issues/408

            resource_metadata_entry = metadata._current_metadata.elements[data_resource['resID']]
            resource_row_elements_evolver = resource_metadata_entry.elements.evolver()
            resource_row_elements_evolver._reallocate(2 * len(media_types))
            for i, media_type in enumerate(media_types):
                column_metadata_entry = metadata_base.MetadataEntry(
                    metadata=frozendict.FrozenOrderedDict({
                        # A media type of this particular file.
                        'media_types': (media_type,),
                    }),
                    is_empty=False,
                )

                row_metadata_entry = metadata_base.MetadataEntry(
                    elements=utils.EMPTY_PMAP.set(0, column_metadata_entry),
                    is_empty=False,
                    is_elements_empty=False,
                )

                resource_row_elements_evolver.set(i, row_metadata_entry)

            resource_metadata_entry.elements = resource_row_elements_evolver.persistent()
            resource_metadata_entry.is_elements_empty = not resource_metadata_entry.elements
            resource_metadata_entry.update_is_empty()

        return data, metadata

    def _load_resource_type_table(self, dataset_path: str, data_resource: typing.Dict, metadata: metadata_base.DataMetadata,
                                  hash: typing.Any) -> typing.Tuple[container_pandas.DataFrame, metadata_base.DataMetadata]:
        assert not data_resource.get('isCollection', False)

        data = None
        column_names = None
        data_path = os.path.join(dataset_path, data_resource['resPath'])

        if utils.is_sequence(data_resource['resFormat']) and len(data_resource['resFormat']) == 1:
            resource_format = data_resource['resFormat'][0]
        elif isinstance(data_resource['resFormat'], typing.Mapping) and len(data_resource['resFormat']) == 1:
            resource_format = list(data_resource['resFormat'].keys())[0]
        else:
            resource_format = None

        if resource_format in ['text/csv', 'text/csv+gzip']:
            data = pandas.read_csv(
                data_path,
                # We do not want to do any conversion of values at this point.
                # This should be done by primitives later on.
                dtype=str,
                # We always expect one row header.
                header=0,
                # We want empty strings and not NaNs.
                na_filter=False,
                compression='gzip' if resource_format == 'text/csv+gzip' else None,
                encoding='utf8',
                low_memory=False,
                memory_map=True,
            )

            column_names = list(data.columns)

            if data_resource.get('columnsCount', None) is not None and len(column_names) != data_resource['columnsCount']:
                raise ValueError("Mismatch between columns count in data {data_count} and expected count {expected_count}.".format(
                    data_count=len(column_names),
                    expected_count=data_resource['columnsCount'],
                ))

            if hash is not None:
                # We include both the filename and the content.
                # TODO: Currently we read the file twice, once for reading and once to compute digest. Could we do it in one pass? Would it make it faster?
                hash.update(data_resource['resPath'].encode('utf8'))
                update_digest(hash, data_path)

        else:
            raise exceptions.NotSupportedError("Resource format '{resource_format}' for table '{resource_path}' is not supported.".format(
                resource_format=data_resource['resFormat'],
                resource_path=data_resource['resPath'],
            ))

        if data is None:
            raise FileNotFoundError("Data file for table '{resource_path}' cannot be found.".format(
                resource_path=data_resource['resPath'],
            ))

        data = container_pandas.DataFrame(data)

        assert column_names is not None

        semantic_types = [D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[data_resource['resType']]]

        if data_resource['resID'] == 'learningData':
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        metadata = metadata.update((data_resource['resID'],), {
            'structural_type': type(data),
            'semantic_types': semantic_types,
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': len(data),
            },
        })

        metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': len(column_names),
            },
        })

        for i, column_name in enumerate(column_names):
            metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS, i), {
                'name': column_name,
                'structural_type': str,
            })

        metadata_columns = {}
        for column in data_resource.get('columns', []):
            metadata_columns[column['colIndex']] = column

        for i in range(len(column_names)):
            if i in metadata_columns:
                if column_names[i] != metadata_columns[i]['colName']:
                    raise ValueError("Mismatch between column name in data '{data_name}' and column name in metadata '{metadata_name}'.".format(
                        data_name=column_names[i],
                        metadata_name=metadata_columns[i]['colName'],
                    ))

                column_metadata = self._get_column_metadata(metadata_columns[i])
            else:
                column_metadata = {
                    'semantic_types': [
                        D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES['unknown'],
                    ],
                }

            if 'https://metadata.datadrivendiscovery.org/types/Boundary' in column_metadata['semantic_types'] and 'boundary_for' not in column_metadata:
                # Let's reconstruct for which column this is a boundary: currently
                # this seems to be the first non-boundary column before this one.
                for column_index in range(i - 1, 0, -1):
                    column_semantic_types = metadata.query((data_resource['resID'], metadata_base.ALL_ELEMENTS, column_index)).get('semantic_types', ())
                    if 'https://metadata.datadrivendiscovery.org/types/Boundary' not in column_semantic_types:
                        column_metadata['boundary_for'] = {
                            'resource_id': data_resource['resID'],
                            'column_index': column_index,
                        }
                        break

            metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS, i), column_metadata)

        current_boundary_start = None
        current_boundary_list: typing.Tuple[str, ...] = None
        column_index = 0
        while column_index < len(column_names):
            column_semantic_types = metadata.query((data_resource['resID'], metadata_base.ALL_ELEMENTS, column_index)).get('semantic_types', ())
            if is_simple_boundary(column_semantic_types):
                # Let's reconstruct which type of a boundary this is. Heuristic is simple.
                # If there are two boundary columns next to each other, it is an interval.
                if current_boundary_start is None:
                    assert current_boundary_list is None

                    count = 1
                    for next_column_index in range(column_index + 1, len(column_names)):
                        if is_simple_boundary(metadata.query((data_resource['resID'], metadata_base.ALL_ELEMENTS, next_column_index)).get('semantic_types', ())):
                            count += 1
                        else:
                            break

                    if count == 2:
                        current_boundary_start = column_index
                        current_boundary_list = INTERVAL_SEMANTIC_TYPES
                    else:
                        # Unsupported group of boundary columns, let's skip them all.
                        column_index += count
                        continue

                column_semantic_types = column_semantic_types + (current_boundary_list[column_index - current_boundary_start],)
                metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS, column_index), {
                    'semantic_types': column_semantic_types,
                })

                if column_index - current_boundary_start + 1 == len(current_boundary_list):
                    current_boundary_start = None
                    current_boundary_list = None

            column_index += 1

        return data, metadata

    def _load_resource_type_edgeList(self, dataset_path: str, data_resource: typing.Dict, metadata: metadata_base.DataMetadata,
                                     hash: typing.Any) -> typing.Tuple[container_pandas.DataFrame, metadata_base.DataMetadata]:
        assert not data_resource.get('isCollection', False)

        return self._load_resource_type_table(dataset_path, data_resource, metadata, hash)

    def _load_resource_type_graph(
        self, dataset_path: str, data_resource: typing.Dict, metadata: metadata_base.DataMetadata, hash: typing.Any,
    ) -> typing.Tuple[container_pandas.DataFrame, metadata_base.DataMetadata]:
        assert not data_resource.get('isCollection', False)

        data_path = os.path.join(dataset_path, data_resource['resPath'])
        collection_path = os.path.dirname(data_path)
        filename = os.path.basename(data_path)
        filename_extension = os.path.splitext(filename)[1]

        try:
            media_type = FILE_EXTENSIONS[filename_extension]
        except KeyError as error:
            raise TypeError("Unsupported file extension for file '{filename}'.".format(filename=filename)) from error

        if hash is not None:
            # We include both the filename and the content.
            hash.update(data_resource['resPath'].encode('utf8'))
            update_digest(hash, data_path)

        data = container_pandas.DataFrame({'filename': [filename]}, columns=['filename'], dtype=object)

        metadata = metadata.update((data_resource['resID'],), {
            'structural_type': type(data),
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/FilesCollection',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': len(data),
            },
        })

        metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 1,
            },
        })

        location_base_uri = utils.fix_uri(collection_path)
        # We want to make sure you can just concat with the filename.
        if not location_base_uri.endswith('/'):
            location_base_uri += '/'

        column_metadata = {
            'name': 'filename',
            'structural_type': str,
            'location_base_uris': [
                location_base_uri,
            ],
            'media_types': [media_type],
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                'https://metadata.datadrivendiscovery.org/types/FileName',
                D3M_RESOURCE_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[data_resource['resType']],
            ],
        }

        metadata = metadata.update((data_resource['resID'], metadata_base.ALL_ELEMENTS, 0), column_metadata)

        return data, metadata

    def _get_column_metadata(self, column: typing.Dict) -> typing.Dict:
        semantic_types = [D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[column['colType']]]

        for role in column['role']:
            semantic_types.append(D3M_ROLE_CONSTANTS_TO_SEMANTIC_TYPES[role])

        # Suggested target is an attribute by default.
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types and 'https://metadata.datadrivendiscovery.org/types/Attribute' not in semantic_types:
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/Attribute')

        # Suggested privileged data is an attribute by default.
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData' in semantic_types and 'https://metadata.datadrivendiscovery.org/types/Attribute' not in semantic_types:
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/Attribute')

        column_metadata: typing.Dict[str, typing.Any] = {
            'semantic_types': semantic_types,
        }

        if column.get('colDescription', None):
            column_metadata['description'] = column['colDescription']

        if column.get('refersTo', None):
            if isinstance(column['refersTo']['resObject'], str):
                if column['refersTo']['resObject'] == 'item':
                    # We represent collections as a table with one column of filenames.
                    column_metadata['foreign_key'] = {
                        'type': 'COLUMN',
                        'resource_id': column['refersTo']['resID'],
                        'column_index': 0,
                    }
                # Legacy (before v4.0.0) node reference.
                elif column['refersTo']['resObject'] == 'node':
                    column_metadata['foreign_key'] = {
                        'type': 'NODE_ATTRIBUTE',
                        'resource_id': column['refersTo']['resID'],
                        'node_attribute': 'nodeID',
                    }
                # Legacy (before v4.0.0) edge reference.
                elif column['refersTo']['resObject'] == 'edge':
                    column_metadata['foreign_key'] = {
                        'type': 'EDGE_ATTRIBUTE',
                        'resource_id': column['refersTo']['resID'],
                        'edge_attribute': 'edgeID',
                    }
                else:
                    raise exceptions.UnexpectedValueError("Unknown \"resObject\" value: {resource_object}".format(resource_object=column['refersTo']['resObject']))
            else:
                if 'columnIndex' in column['refersTo']['resObject']:
                    if 'https://metadata.datadrivendiscovery.org/types/Boundary' in semantic_types:
                        column_metadata['boundary_for'] = {
                            'resource_id': column['refersTo']['resID'],
                            'column_index': column['refersTo']['resObject']['columnIndex'],
                        }
                    else:
                        column_metadata['foreign_key'] = {
                            'type': 'COLUMN',
                            'resource_id': column['refersTo']['resID'],
                            'column_index': column['refersTo']['resObject']['columnIndex'],
                        }
                elif 'columnName' in column['refersTo']['resObject']:
                    if 'https://metadata.datadrivendiscovery.org/types/Boundary' in semantic_types:
                        column_metadata['boundary_for'] = {
                            'resource_id': column['refersTo']['resID'],
                            'column_name': column['refersTo']['resObject']['columnName'],
                        }
                    else:
                        column_metadata['foreign_key'] = {
                            'type': 'COLUMN',
                            'resource_id': column['refersTo']['resID'],
                            'column_name': column['refersTo']['resObject']['columnName'],
                        }
                elif 'nodeAttribute' in column['refersTo']['resObject']:
                    column_metadata['foreign_key'] = {
                        'type': 'NODE_ATTRIBUTE',
                        'resource_id': column['refersTo']['resID'],
                        'node_attribute': column['refersTo']['resObject']['nodeAttribute'],
                    }
                elif 'edgeAttribute' in column['refersTo']['resObject']:
                    column_metadata['foreign_key'] = {
                        'type': 'EDGE_ATTRIBUTE',
                        'resource_id': column['refersTo']['resID'],
                        'edge_attribute': column['refersTo']['resObject']['edgeAttribute'],
                    }
                else:
                    raise exceptions.UnexpectedValueError("Unknown \"resObject\" value: {resource_object}".format(resource_object=column['refersTo']['resObject']))

        if column.get('timeGranularity', None):
            # "units" is backwards compatible field name.
            # See: https://gitlab.com/datadrivendiscovery/data-supply/issues/215
            unit = column['timeGranularity'].get('unit', column['timeGranularity'].get('units', None))
            column_metadata['time_granularity'] = {
                'value': column['timeGranularity']['value'],
                'unit': TIME_GRANULARITIES[unit],
            }

        return column_metadata

    def _merge_score_targets(self, resources: typing.Dict, metadata: metadata_base.DataMetadata, dataset_path: str, hash: typing.Any) -> None:
        targets_path = os.path.join(dataset_path, '..', 'targets.csv')

        targets = pandas.read_csv(
            targets_path,
            # We do not want to do any conversion of values at this point.
            # This should be done by primitives later on.
            dtype=str,
            # We always expect one row header.
            header=0,
            # We want empty strings and not NaNs.
            na_filter=False,
            encoding='utf8',
            low_memory=False,
            memory_map=True,
        )

        for resource_id, resource in resources.items():
            # We assume targets are only in the dataset entry point.
            if metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'):
                contains_empty_values = {}
                for column_name in targets.columns:
                    if column_name == 'd3mIndex':
                        continue

                    contains_empty_values[column_name] = targets.loc[:, column_name].eq('').any()

                # We first make sure targets match resource in row order. At this stage all values
                # are strings, so we can fill simply with empty strings if it happens that index
                # values do not match (which in fact should never happen).
                reindexed_targets = targets.set_index('d3mIndex').reindex(resource.loc[:, 'd3mIndex'], fill_value='').reset_index()

                for column_name in reindexed_targets.columns:
                    if column_name == 'd3mIndex':
                        continue

                    # We match columns based on their names.
                    if column_name in resource.columns:
                        if not contains_empty_values[column_name] and reindexed_targets.loc[:, column_name].eq('').any():
                            raise exceptions.InvalidDatasetError("'d3mIndex' in 'targets.csv' does not match 'd3mIndex' in the resource '{resource_id}'.".format(resource_id=resource_id))

                        resource.loc[:, column_name] = reindexed_targets.loc[:, column_name]

                resources[resource_id] = resource


class CSVLoader(Loader):
    """
    A class for loading a dataset from a CSV file.

    Loader supports both loading a dataset from a local file system or remote locations.
    URI should point to a file with ``.csv`` file extension.
    """

    def can_load(self, dataset_uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(dataset_uri, allow_fragments=False)
        except Exception:
            return False

        if parsed_uri.scheme not in pandas_io_common._VALID_URLS:
            return False

        if parsed_uri.scheme == 'file':
            if parsed_uri.netloc not in ['', 'localhost']:
                return False

            if not parsed_uri.path.startswith('/'):
                return False

        for extension in ('', '.gz', '.bz2', '.zip', 'xz'):
            if parsed_uri.path.endswith('.csv' + extension):
                return True

        return False

    def _load_data(self, resources: typing.Dict, metadata: metadata_base.DataMetadata, *, dataset_uri: str,
                   compute_digest: ComputeDigest) -> typing.Tuple[metadata_base.DataMetadata, int, typing.Optional[str]]:
        try:
            buffer, compression, should_close = self._get_buffer_and_compression(dataset_uri)
        except FileNotFoundError as error:
            raise exceptions.DatasetNotFoundError("CSV dataset '{dataset_uri}' cannot be found.".format(dataset_uri=dataset_uri)) from error
        except urllib_error.HTTPError as error:
            if error.code == 404:
                raise exceptions.DatasetNotFoundError("CSV dataset '{dataset_uri}' cannot be found.".format(dataset_uri=dataset_uri)) from error
            else:
                raise error
        except urllib_error.URLError as error:
            if isinstance(error.reason, FileNotFoundError):
                raise exceptions.DatasetNotFoundError("CSV dataset '{dataset_uri}' cannot be found.".format(dataset_uri=dataset_uri)) from error
            else:
                raise error

        # CSV files do not have digest, so "ALWAYS" and "ONLY_IF_MISSING" is the same.
        # Allowing "True" for backwards compatibility.
        if compute_digest is True or compute_digest == ComputeDigest.ALWAYS or compute_digest == ComputeDigest.ONLY_IF_MISSING:
            buffer_digest = self._get_digest(buffer)
        else:
            buffer_digest = None

        buffer_size = len(buffer.getvalue())

        data = pandas.read_csv(
            buffer,
            # We do not want to do any conversion of values at this point.
            # This should be done by primitives later on.
            dtype=str,
            # We always expect one row header.
            header=0,
            # We want empty strings and not NaNs.
            na_filter=False,
            compression=compression,
            encoding='utf8',
            low_memory=False,
        )

        if should_close:
            try:
                buffer.close()
            except Exception:
                pass

        if 'd3mIndex' not in data.columns:
            # We do not update digest with new data generated here. This is OK because this data is determined by
            # original data so original digest still applies. When saving a new digest has to be computed anyway
            # because this data will have to be converted to string.
            data.insert(0, 'd3mIndex', range(len(data)))
            d3m_index_generated = True
        else:
            d3m_index_generated = False

        data = container_pandas.DataFrame(data)

        resources['learningData'] = data

        metadata = metadata.update(('learningData',), {
            'structural_type': type(data),
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': len(data),
            },
        })

        metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': len(data.columns),
            },
        })

        for i, column_name in enumerate(data.columns):
            if i == 0 and d3m_index_generated:
                metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, i), {
                    'name': column_name,
                    'structural_type': numpy.int64,
                    'semantic_types': [
                        'http://schema.org/Integer',
                        'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                    ],
                })
            else:
                metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, i), {
                    'name': column_name,
                    'structural_type': str,
                    'semantic_types': [
                        'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    ],
                })

        return metadata, buffer_size, buffer_digest

    def _get_buffer_and_compression(self, dataset_uri: str) -> typing.Tuple[io.BytesIO, str, bool]:
        if hasattr(pandas_io_common, 'infer_compression'):
            infer_compression = pandas_io_common.infer_compression
        else:
            # Backwards compatibility for Pandas before 1.0.0.
            infer_compression = pandas_io_common._infer_compression
        compression = infer_compression(dataset_uri, 'infer')
        buffer, _, compression, should_close = pandas_io_common.get_filepath_or_buffer(dataset_uri, 'utf8', compression)

        return buffer, compression, should_close

    def _get_digest(self, buffer: io.BytesIO) -> str:
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    # "strict_digest" is ignored, there is no metadata to compare digest against.
    # "handle_score_split" is ignored as well.
    def load(self, dataset_uri: str, *, dataset_id: str = None, dataset_version: str = None, dataset_name: str = None, lazy: bool = False,
             compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False, handle_score_split: bool = True) -> 'Dataset':
        assert self.can_load(dataset_uri)

        parsed_uri = url_parse.urlparse(dataset_uri, allow_fragments=False)

        # Pandas requires a host for "file" URIs.
        if parsed_uri.scheme == 'file' and parsed_uri.netloc == '':
            parsed_uri = parsed_uri._replace(netloc='localhost')
            dataset_uri = url_parse.urlunparse(parsed_uri)

        dataset_size = None
        dataset_digest = None

        resources: typing.Dict = {}
        metadata = metadata_base.DataMetadata()

        if not lazy:
            load_lazy = None

            metadata, dataset_size, dataset_digest = self._load_data(
                resources, metadata, dataset_uri=dataset_uri, compute_digest=compute_digest,
            )

        else:
            def load_lazy(dataset: Dataset) -> None:
                # "dataset" can be used as "resources", it is a dict of values.
                dataset.metadata, dataset_size, dataset_digest = self._load_data(
                    dataset, dataset.metadata, dataset_uri=dataset_uri, compute_digest=compute_digest,
                )

                new_metadata = {
                    'dimension': {'length': len(dataset)},
                    'stored_size': dataset_size,
                }

                if dataset_digest is not None:
                    new_metadata['digest'] = dataset_digest

                dataset.metadata = dataset.metadata.update((), new_metadata)
                dataset.metadata = dataset.metadata.generate(dataset)

                dataset._load_lazy = None

        dataset_metadata = {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': Dataset,
            'id': dataset_id or dataset_uri,
            'name': dataset_name or os.path.basename(parsed_uri.path),
            'location_uris': [
                dataset_uri,
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                'length': len(resources),
            },
        }

        if dataset_version is not None:
            dataset_metadata['version'] = dataset_version

        if dataset_size is not None:
            dataset_metadata['stored_size'] = dataset_size

        if dataset_digest is not None:
            dataset_metadata['digest'] = dataset_digest

        metadata = metadata.update((), dataset_metadata)

        return Dataset(resources, metadata, load_lazy=load_lazy)


class SklearnExampleLoader(Loader):
    """
    A class for loading example scikit-learn datasets.

    URI should be of the form ``sklearn://<name of the dataset>``, where names come from
    ``sklearn.datasets.load_*`` function names.
    """

    def can_load(self, dataset_uri: str) -> bool:
        if dataset_uri.startswith('sklearn://'):
            return True

        return False

    def _load_data(self, resources: typing.Dict, metadata: metadata_base.DataMetadata, *, dataset_path: str,
                   compute_digest: ComputeDigest) -> typing.Tuple[metadata_base.DataMetadata, typing.Optional[str], typing.Optional[str]]:
        bunch = self._get_bunch(dataset_path)

        # Sklearn datasets do not have digest, so "ALWAYS" and "ONLY_IF_MISSING" is the same.
        # Allowing "True" for backwards compatibility.
        if compute_digest is True or compute_digest == ComputeDigest.ALWAYS or compute_digest == ComputeDigest.ONLY_IF_MISSING:
            bunch_digest = self._get_digest(bunch)
        else:
            bunch_digest = None

        bunch_description = bunch.get('DESCR', None) or None

        bunch_data = bunch['data']
        bunch_target = bunch['target']

        if len(bunch_data.shape) == 1:
            bunch_data = bunch_data.reshape((bunch_data.shape[0], 1))
        if len(bunch_target.shape) == 1:
            bunch_target = bunch_target.reshape((bunch_target.shape[0], 1))

        column_names = []
        target_values = None

        if 'feature_names' in bunch:
            for feature_name in bunch['feature_names']:
                column_names.append(str(feature_name))

        if 'target_names' in bunch:
            if len(bunch['target_names']) == bunch_target.shape[1]:
                for target_name in bunch['target_names']:
                    column_names.append(str(target_name))
            else:
                target_values = [str(target_value) for target_value in bunch['target_names']]

        if target_values is not None:
            converted_target = numpy.empty(bunch_target.shape, dtype=object)

            for i, row in enumerate(bunch_target):
                for j, column in enumerate(row):
                    converted_target[i, j] = target_values[column]
        else:
            converted_target = bunch_target

        # Add names for any extra columns. We do not really check for duplicates because Pandas allow columns with the same name.
        for i in range(len(column_names), bunch_data.shape[1] + converted_target.shape[1]):
            column_names.append('column {i}'.format(i=i))

        data = pandas.concat([pandas.DataFrame(bunch_data), pandas.DataFrame(converted_target)], axis=1)
        data.columns = column_names
        data = container_pandas.DataFrame(data)

        # We do not update digest with new data generated here. This is OK because this data is determined by
        # original data so original digest still applies. When saving a new digest has to be computed anyway
        # because this data will have to be converted to string.
        data.insert(0, 'd3mIndex', range(len(data)))

        resources['learningData'] = data

        metadata = metadata.update(('learningData',), {
            'structural_type': type(data),
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': len(data),
            },
        })

        metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': len(data.columns),
            },
        })

        metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 0), {
            'name': 'd3mIndex',
            'structural_type': numpy.int64,
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        for column_index in range(1, bunch_data.shape[1] + 1):
            column_metadata: typing.Dict[str, typing.Any] = {
                'structural_type': bunch_data.dtype.type,
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
                'name': data.columns[column_index],
            }

            metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, column_index), column_metadata)

        for column_index in range(bunch_data.shape[1] + 1, bunch_data.shape[1] + bunch_target.shape[1] + 1):
            if target_values is not None:
                if len(target_values) == 2:
                    column_type = ['http://schema.org/Boolean']
                elif len(target_values) > 2:
                    column_type = ['https://metadata.datadrivendiscovery.org/types/CategoricalData']
                else:
                    raise exceptions.InvalidDatasetError("Too few target values in sklearn dataset.")
            else:
                column_type = ['https://metadata.datadrivendiscovery.org/types/UnknownType']

            column_metadata = {
                'structural_type': str if target_values is not None else bunch_target.dtype.type,
                'semantic_types': column_type + [
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
                'name': data.columns[column_index],
            }

            metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, column_index), column_metadata)

        return metadata, bunch_description, bunch_digest

    def _get_digest(self, bunch: typing.Dict) -> str:
        hash = hashlib.sha256()

        hash.update(bunch['data'].tobytes())
        hash.update(bunch['target'].tobytes())

        if 'feature_names' in bunch:
            if isinstance(bunch['feature_names'], list):
                for feature_name in bunch['feature_names']:
                    hash.update(feature_name.encode('utf8'))
            else:
                hash.update(bunch['feature_names'].tobytes())

        if 'target_names' in bunch:
            if isinstance(bunch['target_names'], list):
                for target_name in bunch['target_names']:
                    hash.update(target_name.encode('utf8'))
            else:
                hash.update(bunch['target_names'].tobytes())

        if 'DESCR' in bunch:
            hash.update(bunch['DESCR'].encode('utf8'))

        return hash.hexdigest()

    def _get_bunch(self, dataset_path: str) -> typing.Dict:
        return getattr(datasets, 'load_{dataset_path}'.format(dataset_path=dataset_path))()

    # "strict_digest" is ignored, there is no metadata to compare digest against.
    # "handle_score_split is ignored as well.
    def load(self, dataset_uri: str, *, dataset_id: str = None, dataset_version: str = None, dataset_name: str = None, lazy: bool = False,
             compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False, handle_score_split: bool = True) -> 'Dataset':
        assert self.can_load(dataset_uri)

        dataset_path = dataset_uri[len('sklearn://'):]

        if not hasattr(datasets, 'load_{dataset_path}'.format(dataset_path=dataset_path)):
            raise exceptions.DatasetNotFoundError("Sklearn dataset '{dataset_uri}' cannot be found.".format(dataset_uri=dataset_uri))

        dataset_description = None
        dataset_digest = None

        resources: typing.Dict = {}
        metadata = metadata_base.DataMetadata()

        if not lazy:
            load_lazy = None

            metadata, dataset_description, dataset_digest = self._load_data(
                resources, metadata, dataset_path=dataset_path, compute_digest=compute_digest,
            )

        else:
            def load_lazy(dataset: Dataset) -> None:
                # "dataset" can be used as "resources", it is a dict of values.
                dataset.metadata, dataset_description, dataset_digest = self._load_data(
                    dataset, dataset.metadata, dataset_path=dataset_path, compute_digest=compute_digest,
                )

                new_metadata: typing.Dict = {
                    'dimension': {'length': len(dataset)},
                }

                if dataset_description is not None:
                    new_metadata['description'] = dataset_description

                if dataset_digest is not None:
                    new_metadata['digest'] = dataset_digest

                dataset.metadata = dataset.metadata.update((), new_metadata)
                dataset.metadata = dataset.metadata.generate(dataset)

                dataset._load_lazy = None

        dataset_metadata = {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': Dataset,
            'id': dataset_id or dataset_uri,
            'name': dataset_name or dataset_path,
            'location_uris': [
                dataset_uri,
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                'length': len(resources),
            },
        }

        if dataset_version is not None:
            dataset_metadata['version'] = dataset_version

        if dataset_description is not None:
            dataset_metadata['description'] = dataset_description

        if dataset_digest is not None:
            dataset_metadata['digest'] = dataset_digest

        metadata = metadata.update((), dataset_metadata)

        return Dataset(resources, metadata, load_lazy=load_lazy)


class D3MDatasetSaver(Saver):
    """
    A class for saving of D3M datasets.

    This saver supports only saving to local file system.
    URI should point to the ``datasetDoc.json`` file in the D3M dataset directory.
    """

    VERSION = '4.1.0'

    def can_save(self, dataset_uri: str) -> bool:
        if not self._is_dataset(dataset_uri):
            return False

        if not self._is_local_file(dataset_uri):
            return False

        return True

    def _is_dataset(self, uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(uri, allow_fragments=False)
        except Exception:
            return False

        if os.path.basename(parsed_uri.path) != 'datasetDoc.json':
            return False

        return True

    def _is_local_file(self, uri: str) -> bool:
        try:
            parsed_uri = url_parse.urlparse(uri, allow_fragments=False)
        except Exception:
            return False

        if parsed_uri.scheme != 'file':
            return False

        if parsed_uri.netloc not in ['', 'localhost']:
            return False

        if not parsed_uri.path.startswith('/'):
            return False

        return True

    def _get_column_description(self, column_index: int, column_name: str, column_metadata: typing.Dict) -> typing.Dict:
        column = {
            'colIndex': column_index,
            'colName': column_name,
            'role': [SEMANTIC_TYPES_TO_D3M_ROLES[x] for x in column_metadata.get('semantic_types', []) if x in SEMANTIC_TYPES_TO_D3M_ROLES]
        }
        column_type = [SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES[semantic_type] for semantic_type in column_metadata.get('semantic_types', []) if semantic_type in SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES]

        # If column semantic_type is not specified we default to unknown type.
        if not column_type:
            if 'structural_type' in column_metadata:
                if utils.is_int(column_metadata['structural_type']):
                    column['colType'] = SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES['http://schema.org/Integer']
                elif utils.is_float(column_metadata['structural_type']):
                    column['colType'] = SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES['http://schema.org/Float']
                elif issubclass(column_metadata['structural_type'], bool):
                    column['colType'] = SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES['http://schema.org/Boolean']
                else:
                    column['colType'] = SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES['https://metadata.datadrivendiscovery.org/types/UnknownType']
            else:
                column['colType'] = SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES['https://metadata.datadrivendiscovery.org/types/UnknownType']
        elif len(column_type) == 1:
            column['colType'] = column_type[0]
        else:
            raise exceptions.InvalidMetadataError(
                "More than one semantic type found for column type: {column_type}".format(
                    column_type=column_type,
                ),
            )

        if column_metadata.get('description', None):
            column['colDescription'] = column_metadata['description']

        return column

    def _get_collection_resource_description(self, dataset: 'Dataset', resource_id: str, resource: typing.Any, dataset_location_base_path: typing.Optional[str]) -> typing.Dict:
        if not isinstance(resource, container_pandas.DataFrame):
            raise exceptions.InvalidArgumentTypeError("Saving a D3M dataset with a collection resource which is not a DataFrame, but '{structural_type}'.".format(
                structural_type=type(resource),
            ))
        if len(resource.columns) != 1:
            raise exceptions.InvalidArgumentTypeError("Saving a D3M dataset with a collection resource with an invalid number of columns: {columns}".format(
                columns=len(resource.columns),
            ))
        if not dataset.metadata.has_semantic_type((resource_id, metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/FileName'):
            raise exceptions.InvalidArgumentTypeError("Saving a D3M dataset with a collection resource with with a column which does not contain filenames.")

        selector = (resource_id, metadata_base.ALL_ELEMENTS, 0)
        metadata, exceptions_with_selectors = dataset.metadata.query_with_exceptions(selector)

        # We check structural type for all rows in a column, but also if any row has a different structural type.
        for structural_type in [metadata['structural_type']] + [metadata['structural_type'] for metadata in exceptions_with_selectors.values() if 'structural_type' in metadata]:
            if not issubclass(structural_type, str):
                raise exceptions.InvalidArgumentTypeError("Saving a D3M dataset with a collection resource with with a column which does not just string values, but also '{structural_type}'.".format(
                    structural_type=structural_type,
                ))

        # We use "location_base_uris" from all rows. We only support "location_base_uris"
        # being the same for all rows, so we have to verify that.
        all_location_base_uris_nested = [
            list(metadata.get('location_base_uris', []))
        ] + [
            list(metadata['location_base_uris']) for metadata in exceptions_with_selectors.values() if 'location_base_uris' in metadata
        ]

        # Flatten the list of lists, remove duplicates, sort for reproducibility.
        all_location_base_uris = sorted({all_location_base_uri for all_location_base_uri in itertools.chain.from_iterable(all_location_base_uris_nested)})

        local_location_base_uris = [location_base_uri for location_base_uri in all_location_base_uris if self._is_local_file(location_base_uri)]

        if not local_location_base_uris:
            raise exceptions.NotSupportedError(
                "Saving a D3M dataset with a collection resource without local files is not supported: {all_location_base_uris}".format(
                    all_location_base_uris=all_location_base_uris,
                ),
            )
        elif len(local_location_base_uris) > 1:
            # When there are multiple base locations in D3M dataset format can lead to conflicts
            # where same filename in a column points to different files, but we are storing them
            # under the same resource path. We verify that there are no conflicts in "_save_collection".
            # Because there is no clear way to determine the best common resource path we use a hard-coded one.
            resource_path = 'files/'
        elif dataset_location_base_path is None:
            # We cannot determine the resource path so we use a hard-coded one.
            resource_path = 'files/'
        else:
            location_base_path = url_parse.urlparse(local_location_base_uris[0], allow_fragments=False).path

            # This is a way to check that "dataset_location_base_path" is a prefix of "location_base_path".
            if os.path.commonpath([location_base_path, dataset_location_base_path]) != dataset_location_base_path:
                raise exceptions.NotSupportedError(
                    "Saving a D3M dataset with a collection resource with files location not under the dataset directory.",
                )

            resource_path = location_base_path[len(dataset_location_base_path) + 1:]

        # Just a matter of style.
        if not resource_path.endswith('/'):
            resource_path += '/'

        resource_formats_set = set()
        # "media_types" for "ALL_ELEMENTS" is an union of all rows.
        for media_type in metadata.get('media_types', []):
            # We allow unknown media types.
            resource_formats_set.add(MEDIA_TYPES_REVERSE.get(media_type, media_type))

        resource_formats = {}

        # An empty collection? Or just a collection resource without metadata?
        if not resource_formats_set:
            if len(resource):
                raise ValueError("A collection resource without media types metadata.")

        # An optimized case, all files in a collection belong to the same resource format.
        elif len(resource_formats_set) == 1:
            file_extensions_set = set()
            for filename in resource.iloc[:, 0]:
                root, ext = os.path.splitext(filename)
                if not ext:
                    raise ValueError("A filename without a file extension in a collection resource: {filename}".format(filename=filename))
                ext = _remove_extension_dot(ext)
                file_extensions_set.add(ext)

            # Sorting to have reproducibility.
            resource_formats[resource_formats_set.pop()] = sorted(file_extensions_set)

        else:
            resource_formats_of_sets: typing.Dict[str, typing.Set] = {}

            for row_index, filename in enumerate(resource.iloc[:, 0]):
                root, ext = os.path.splitext(filename)
                if not ext:
                    raise ValueError("A filename without a file extension in a collection resource: {filename}".format(filename=filename))
                ext = _remove_extension_dot(ext)

                try:
                    media_types = dataset.metadata.query((resource_id, row_index, 0))['media_types']
                except KeyError:
                    raise ValueError("A collection resource without media types metadata for row {row_index}.".format(row_index=row_index)) from None

                if len(media_types) != 1:
                    raise ValueError("Medata should have only one media type per row in a collection resource, at row {row_index}: {media_types}".format(row_index=row_index, media_types=media_types))

                # We allow unknown media types.
                resource_format = MEDIA_TYPES_REVERSE.get(media_types[0], media_types[0])

                if resource_format not in resource_formats_of_sets:
                    resource_formats_of_sets[resource_format] = set()

                resource_formats_of_sets[resource_format].add(ext)

            for resource_format, file_extensions in resource_formats_of_sets.items():
                # Sorting to have reproducibility.
                resource_formats[resource_format] = sorted(file_extensions)

        resource_type = [SEMANTIC_TYPES_TO_D3M_RESOURCE_TYPES[semantic_type] for semantic_type in metadata.get('semantic_types', []) if semantic_type in SEMANTIC_TYPES_TO_D3M_RESOURCE_TYPES]

        if len(resource_type) != 1:
            raise exceptions.InvalidMetadataError(
                "Not exactly one semantic type found for resource type: {resource_type}".format(
                    resource_type=resource_type,
                ),
            )

        resource_description = {
            'resID': resource_id,
            'isCollection': True,
            'resFormat': resource_formats,
            'resType': resource_type[0],
            'resPath': resource_path,
        }

        columns = self._get_columns_description(dataset, resource_id, resource)

        if columns:
            resource_description['columns'] = columns

        if 'file_columns_count' in metadata:
            resource_description['columnsCount'] = metadata['file_columns_count']

        return resource_description

    # We do not use "dataset_location_base_path" but we keep it for all "_get_*_resource_description" methods to have the same signature.
    def _get_dataframe_resource_description(self, dataset: 'Dataset', resource_id: str, resource: typing.Any, dataset_location_base_path: typing.Optional[str]) -> typing.Dict:
        if dataset.metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/EdgeList'):
            res_type = 'edgeList'
        else:
            res_type = 'table'

        resource_description = {
            'resID': resource_id,
            'isCollection': False,
            'resFormat': {'text/csv': ['csv']},
            'resType': res_type,
            'columnsCount': len(resource.columns),
        }

        if dataset.metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'):
            if resource_id != 'learningData':
                logger.error("Saving a dataset with a dataset entry point with resource ID not equal to 'learningData', but '%(resource_id)s'.", {'resource_id': resource_id})
            resource_description['resPath'] = 'tables/learningData.csv'
        else:
            resource_description['resPath'] = 'tables/{resource_id}.csv'.format(resource_id=resource_id)

        columns = self._get_columns_description(dataset, resource_id, resource)

        if columns:
            resource_description['columns'] = columns

        return resource_description

    # TODO: Make it easier to subclass to support other resource types.
    def _get_resource_description(self, dataset: 'Dataset', resource_id: str, resource: typing.Any, dataset_location_base_path: typing.Optional[str]) -> typing.Dict:
        if dataset.metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/FilesCollection'):
            return self._get_collection_resource_description(dataset, resource_id, resource, dataset_location_base_path)

        elif isinstance(resource, container_pandas.DataFrame):
            return self._get_dataframe_resource_description(dataset, resource_id, resource, dataset_location_base_path)

        else:
            raise exceptions.NotSupportedError("Saving a D3M dataset with a resource with structural type '{structural_type}' is not supported.".format(structural_type=type(resource)))

    def _get_columns_description(self, dataset: 'Dataset', resource_id: str, resource: typing.Any) -> typing.List[typing.Dict]:
        columns = []

        # Traverse file columns in collections.
        if dataset.metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/FilesCollection'):
            # We know there is only one column here. This has been verified in "_get_collection_resource_description".
            column_metadata = dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS, 0))
            for file_column_metadata in column_metadata.get('file_columns', []):
                columns.append(self._get_column_description(file_column_metadata['column_index'], file_column_metadata['column_name'], file_column_metadata))

        # Traverse columns in a DataFrame.
        elif isinstance(resource, container_pandas.DataFrame):
            number_of_columns = len(resource.columns)
            for column_index in range(number_of_columns):
                column_selector = (resource_id, metadata_base.ALL_ELEMENTS, column_index)
                column_metadata = dataset.metadata.query(column_selector)

                column = self._get_column_description(column_index, column_metadata['name'], column_metadata)

                if 'boundary_for' in column_metadata and 'foreign_key' in column_metadata:
                    raise exceptions.NotSupportedError("Both boundary and foreign key are not supported.")

                elif 'foreign_key' in column_metadata:
                    if column_metadata['foreign_key']['type'] == 'COLUMN':
                        refers_to = {
                            'resID': column_metadata['foreign_key']['resource_id'],
                            'resObject': {},
                        }

                        if 'column_name' in column_metadata['foreign_key']:
                            refers_to['resObject'] = {
                                'columnName': column_metadata['foreign_key']['column_name'],
                            }
                            referring_column_index = dataset.metadata.get_column_index_from_column_name(
                                column_metadata['foreign_key']['column_name'],
                                at=(column_metadata['foreign_key']['resource_id'],),
                            )
                        elif 'column_index' in column_metadata['foreign_key']:
                            refers_to['resObject'] = {
                                'columnIndex': column_metadata['foreign_key']['column_index'],
                            }
                            referring_column_index = column_metadata['foreign_key']['column_index']
                        else:
                            raise exceptions.InvalidMetadataError(f"'foreign_key' is missing a column reference, in metadata of column {column_index} of resource '{resource_id}'.")

                        # A special case to handle a reference to a file collection.
                        if dataset.metadata.has_semantic_type(
                            (column_metadata['foreign_key']['resource_id'],),
                            'https://metadata.datadrivendiscovery.org/types/FilesCollection',
                        ) and dataset.metadata.has_semantic_type(
                            (column_metadata['foreign_key']['resource_id'], metadata_base.ALL_ELEMENTS, referring_column_index),
                            'https://metadata.datadrivendiscovery.org/types/FileName',
                        ):
                            refers_to['resObject'] = 'item'

                        column['refersTo'] = refers_to

                    elif column_metadata['foreign_key']['type'] == 'NODE_ATTRIBUTE':
                        column['refersTo'] = {
                            'resID': column_metadata['foreign_key']['resource_id'],
                            'resObject': {
                                'nodeAttribute': column_metadata['foreign_key']['node_attribute'],
                            },
                        }

                    elif column_metadata['foreign_key']['type'] == 'EDGE_ATTRIBUTE':
                        column['refersTo'] = {
                            'resID': column_metadata['foreign_key']['resource_id'],
                            'resObject': {
                                'edgeAttribute': column_metadata['foreign_key']['edge_attribute'],
                            },
                        }

                elif 'boundary_for' in column_metadata:
                    refers_to = {
                        # "resource_id" is optional in our metadata and it
                        # means the reference is local to the resource.
                        'resID': column_metadata['boundary_for'].get('resource_id', resource_id),
                        'resObject': {},
                    }

                    if 'column_name' in column_metadata['boundary_for']:
                        refers_to['resObject'] = {
                            'columnName': column_metadata['boundary_for']['column_name'],
                        }
                    elif 'column_index' in column_metadata['boundary_for']:
                        refers_to['resObject'] = {
                            'columnIndex': column_metadata['boundary_for']['column_index'],
                        }
                    else:
                        raise exceptions.InvalidMetadataError(f"'boundary_for' is missing a column reference, in metadata of column {column_index} of resource '{resource_id}'.")

                    column['refersTo'] = refers_to

                if 'time_granularity' in column_metadata:
                    try:
                        column['timeGranularity'] = {
                            'value': column_metadata['time_granularity']['value'],
                            'unit': TIME_GRANULARITIES_REVERSE[column_metadata['time_granularity']['unit']],
                        }
                    except KeyError as error:
                        raise exceptions.InvalidMetadataError(f"'time_granularity' is invalid, in metadata of column {column_index} of resource '{resource_id}'.") from error

                columns.append(column)

        return columns

    def _get_dataset_description(self, dataset: 'Dataset') -> typing.Dict:
        dataset_description: typing.Dict[str, typing.Any] = {
            'about': {
                'datasetSchemaVersion': self.VERSION,
            },
        }

        dataset_root_metadata = dataset.metadata.query(())

        for d3m_path, (dataset_path, required) in D3M_TO_DATASET_FIELDS.items():
            value = utils.get_dict_path(dataset_root_metadata, dataset_path)
            if value is not None:
                utils.set_dict_path(dataset_description, d3m_path, value)
            elif required:
                raise exceptions.InvalidMetadataError(f"Dataset metadata field '{'.'.join(dataset_path)}' is required when saving.")

        for x in [dataset_root_metadata.get('stored_size', None), dataset_description['about'].get('approximateSize', None)]:
            if x is not None:
                exponent = int((math.log10(x) // 3) * 3)
                try:
                    unit = SIZE_TO_UNITS[exponent]
                except KeyError as error:
                    raise KeyError("Unit string for '{exponent}' not found in lookup dictionary {SIZE_TO_UNITS}.".format(exponent=exponent, SIZE_TO_UNITS=SIZE_TO_UNITS)) from error
                dataset_description['about']['approximateSize'] = str(x // (10 ** exponent)) + ' ' + unit
                break

        # We are only using the first URI due to design of D3M dataset format. Remaining URIs should be stored in qualities.
        if dataset_root_metadata.get('source', {}).get('uris', []):
            dataset_description['about']['sourceURI'] = dataset_root_metadata['source']['uris'][0]

        dataset_location_uris = [location_uri for location_uri in dataset_root_metadata.get('location_uris', []) if self._is_local_file(location_uri)]

        if dataset_location_uris:
            # If there are multiple local URIs, we pick the first.
            dataset_location_base_path = os.path.dirname(url_parse.urlparse(dataset_location_uris[0], allow_fragments=False).path)
        else:
            dataset_location_base_path = None

        data_resources = []

        for resource_id, resource in dataset.items():
            resource_description = self._get_resource_description(dataset, resource_id, resource, dataset_location_base_path)

            data_resources.append(resource_description)

        dataset_description['dataResources'] = data_resources

        return dataset_description

    def _generate_metadata_qualities(self, dataset: 'Dataset') -> typing.List:
        # We start with canonical metadata.
        metadata_to_save = dataset._canonical_metadata(dataset.metadata)

        # We remove digest.
        metadata_to_save = metadata_to_save.update((), {'digest': metadata_base.NO_VALUE})

        for resource_id, resource in dataset.items():
            if isinstance(resource, container_pandas.DataFrame):
                # All columns in the DataFrame will be saved as strings, so we have to update
                # metadata first to reflect that, before we save metadata.
                metadata_to_save = metadata_to_save.update((resource_id, metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS), {'structural_type': str})

        qualities = []
        for metadata_entry in metadata_to_save.to_internal_json_structure():
            restricted_to = {
                'resID': metadata_entry['selector'][0] if metadata_entry['selector'] else '',
            }

            if metadata_entry['selector']:
                restricted_to['resComponent'] = {
                    'selector': metadata_entry['selector'][1:],
                }

            qualities.append({
                'qualName': 'metadata',
                'qualValue': metadata_entry['metadata'],
                'qualValueType': 'dict',
                'restrictedTo': restricted_to,
            })

        return qualities

    def save(self, dataset: 'Dataset', dataset_uri: str, *, compute_digest: ComputeDigest = ComputeDigest.ALWAYS, preserve_metadata: bool = True) -> None:
        assert self.can_save(dataset_uri)

        dataset_description = self._get_dataset_description(dataset)

        if preserve_metadata:
            dataset_description['qualities'] = self._generate_metadata_qualities(dataset)

        dataset_path = os.path.dirname(url_parse.urlparse(dataset_uri, allow_fragments=False).path)
        os.makedirs(dataset_path, 0o755, exist_ok=False)

        dataset_description_path = os.path.join(dataset_path, 'datasetDoc.json')

        # We use "x" mode to make sure file does not already exist.
        with open(dataset_description_path, 'x', encoding='utf8') as f:
            json.dump(dataset_description, f, indent=2, allow_nan=False)

        for resource_description in dataset_description['dataResources']:
            resource_id = resource_description['resID']
            resource = dataset[resource_id]

            self._save_resource(dataset, dataset_uri, dataset_path, resource_description, resource_id, resource)

        # We calculate digest of the new dataset and write it into datasetDoc.json
        dataset_description['about']['digest'] = get_d3m_dataset_digest(dataset_description_path)
        with open(dataset_description_path, 'w', encoding='utf8') as f:
            json.dump(dataset_description, f, indent=2, allow_nan=False)

    # TODO: Make it easier to subclass to support non-local "location_base_uris".
    def _save_collection(self, dataset: 'Dataset', dataset_uri: str, dataset_path: str, resource_description: typing.Dict, resource_id: str, resource: typing.Any) -> None:
        # Here we can assume collection resource is a DataFrame which contains exactly one
        # column containing filenames. This has been verified in "_get_collection_resource_description".
        assert isinstance(resource, container_pandas.DataFrame), type(resource)
        assert len(resource.columns) == 1, resource.columns

        already_copied: typing.Set[typing.Tuple[str, str]] = set()
        linking_warning_issued = False

        for row_index, filename in enumerate(resource.iloc[:, 0]):
            # "location_base_uris" is required for collections.
            location_base_uris = dataset.metadata.query((resource_id, row_index, 0))['location_base_uris']

            local_location_base_uris = [location_base_uri for location_base_uri in location_base_uris if self._is_local_file(location_base_uri)]

            # We verified in "_get_collection_resource_description" that there is only one local URI.
            assert len(local_location_base_uris) == 1, local_location_base_uris
            local_location_base_uri = local_location_base_uris[0]

            # "location_base_uris" should be made so that we can just concat with the filename
            # ("location_base_uris" end with "/").
            source_uri = local_location_base_uri + filename
            source_path = url_parse.urlparse(source_uri, allow_fragments=False).path

            destination_path = os.path.join(dataset_path, resource_description['resPath'], filename)

            # Multiple rows can point to the same file, so we do not have to copy them multiple times.
            if (source_path, destination_path) in already_copied:
                continue

            os.makedirs(os.path.dirname(destination_path), 0o755, exist_ok=True)

            linked = False

            try:
                os.link(source_path, destination_path)
                linked = True

            except FileExistsError as error:
                # If existing file is the same, then this is OK. Multiple rows can point to the same file.
                if os.path.samefile(source_path, destination_path):
                    linked = True
                elif filecmp.cmp(source_path, destination_path, shallow=False):
                    linked = True
                # But otherwise we raise an exception.
                else:
                    raise exceptions.AlreadyExistsError(
                        "Destination file '{destination_path}' already exists with different content than '{source_path}' has.".format(
                            destination_path=destination_path,
                            source_path=source_path,
                        ),
                    ) from error

            except OSError as error:
                # OSError: [Errno 18] Invalid cross-device link
                if error.errno == errno.EXDEV:
                    pass
                else:
                    raise error

            # If we can't make a hard-link we try to copy the file.
            if not linked:
                if not linking_warning_issued:
                    linking_warning_issued = True
                    logger.warning("Saving dataset to '%(dataset_uri)s' cannot use hard-linking.", {'dataset_uri': dataset_uri})

                try:
                    with open(source_path, 'rb') as source_file:
                        with open(destination_path, 'xb') as destination_file:
                            shutil.copyfileobj(source_file, destination_file)

                except FileExistsError as error:
                    # If existing file is the same, then this is OK. Multiple rows can point to the same file.
                    if os.path.samefile(source_path, destination_path):
                        pass
                    elif filecmp.cmp(source_path, destination_path, shallow=False):
                        pass
                    # But otherwise we raise an exception.
                    else:
                        raise exceptions.AlreadyExistsError(
                            "Destination file '{destination_path}' already exists with different content than '{source_path}' has.".format(
                                destination_path=destination_path,
                                source_path=source_path,
                            ),
                        ) from error

            already_copied.add((source_path, destination_path))

    # TODO: Make it easier to subclass to support other column types.
    def _save_dataframe(self, dataset: 'Dataset', dataset_path: str, resource_description: typing.Dict, resource_id: str, resource: typing.Any) -> None:
        destination_path = os.path.join(dataset_path, resource_description['resPath'])
        # A subset of "simple_data_types".
        # TODO: Support additional types.
        #       Dicts we can try to convert to "json" column type. Lists of floats we can convert to "realVector".
        #       We could also probably support boolean values.
        supported_column_structural_types = (str, float, int, numpy.integer, numpy.float64, numpy.bool_, type(None))

        # We verify if structural types of columns are supported.
        for column_index in range(dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']):
            selector = (resource_id, metadata_base.ALL_ELEMENTS, column_index)
            metadata, exceptions_with_selectors = dataset.metadata.query_with_exceptions(selector)

            # We check structural type for all rows in a column, but also if any row has a different structural type.
            for structural_type in [metadata['structural_type']] + [metadata['structural_type'] for metadata in exceptions_with_selectors.values() if 'structural_type' in metadata]:
                if not issubclass(structural_type, supported_column_structural_types):
                    raise exceptions.NotSupportedError("Saving a D3M dataset with a column with structural type '{structural_type}' is not supported.".format(structural_type=structural_type))

        os.makedirs(os.path.dirname(destination_path), 0o755, exist_ok=True)

        # We use "x" mode to make sure file does not already exist.
        resource.to_csv(destination_path, mode='x', encoding='utf8')

    # TODO: Make it easier to subclass to support other resource types.
    def _save_resource(self, dataset: 'Dataset', dataset_uri: str, dataset_path: str, resource_description: typing.Dict, resource_id: str, resource: typing.Any) -> None:
        if resource_description.get('isCollection', False):
            self._save_collection(dataset, dataset_uri, dataset_path, resource_description, resource_id, resource)

        elif isinstance(resource, container_pandas.DataFrame):
            self._save_dataframe(dataset, dataset_path, resource_description, resource_id, resource)

        else:
            raise exceptions.NotSupportedError("Saving a D3M dataset with a resource with structural type '{structural_type}' is not supported.".format(structural_type=type(resource)))


D = typing.TypeVar('D', bound='Dataset')


# TODO: It should be probably immutable.
class Dataset(dict):
    """
    A class representing a dataset.

    Internally, it is a dictionary containing multiple resources (e.g., tables).

    Parameters
    ----------
    resources:
        A map from resource IDs to resources.
    metadata:
        Metadata associated with the ``data``.
    load_lazy:
        If constructing a lazy dataset, calling this function will read all the
        data and convert the dataset to a non-lazy one.
    generate_metadata: bool
        Automatically generate and update the metadata.
    check:
        DEPRECATED: argument ignored.
    source:
        DEPRECATED: argument ignored.
    timestamp:
        DEPRECATED: argument ignored.
    """

    metadata: metadata_base.DataMetadata = None
    loaders: typing.List[Loader] = [
        D3MDatasetLoader(),
        CSVLoader(),
        SklearnExampleLoader(),
        OpenMLDatasetLoader(),
    ]
    savers: typing.List[Saver] = [
        D3MDatasetSaver(),
    ]

    @deprecate.arguments('source', 'timestamp', 'check', message="argument ignored")
    def __init__(self, resources: typing.Mapping, metadata: metadata_base.DataMetadata = None, *,
                 load_lazy: typing.Callable[['Dataset'], None] = None, generate_metadata: bool = False,
                 check: bool = True, source: typing.Any = None, timestamp: datetime.datetime = None) -> None:
        super().__init__(resources)

        if isinstance(resources, Dataset) and metadata is None:
            # We made a copy, so we do not have to generate metadata.
            self.metadata = resources.metadata
        elif metadata is not None:
            # We were provided metadata, so we do not have to generate metadata.
            self.metadata = metadata
        else:
            self.metadata = metadata_base.DataMetadata()
            if generate_metadata:
                self.metadata = self.metadata.generate(self)

        self._load_lazy = load_lazy

    @classmethod
    def load(cls, dataset_uri: str, *, dataset_id: str = None, dataset_version: str = None, dataset_name: str = None, lazy: bool = False,
             compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False, handle_score_split: bool = True) -> 'Dataset':
        """
        Tries to load dataset from ``dataset_uri`` using all registered dataset loaders.

        Parameters
        ----------
        dataset_uri:
            A URI to load.
        dataset_id:
            Override dataset ID determined by the loader.
        dataset_version:
            Override dataset version determined by the loader.
        dataset_name:
            Override dataset name determined by the loader.
        lazy:
            If ``True``, load only top-level metadata and not whole dataset.
        compute_digest:
            Compute a digest over the data?
        strict_digest:
            If computed digest does not match the one provided in metadata, raise an exception?
        handle_score_split:
            If a scoring dataset has target values in a separate file, merge them in?

        Returns
        -------
        A loaded dataset.
        """

        for loader in cls.loaders:
            if loader.can_load(dataset_uri):
                return loader.load(
                    dataset_uri, dataset_id=dataset_id, dataset_version=dataset_version, dataset_name=dataset_name,
                    lazy=lazy, compute_digest=compute_digest, strict_digest=strict_digest, handle_score_split=handle_score_split,
                )

        raise exceptions.DatasetUriNotSupportedError(
            "No known loader could load dataset from '{dataset_uri}'.".format(dataset_uri=dataset_uri),
        )

    def save(self, dataset_uri: str, *, compute_digest: ComputeDigest = ComputeDigest.ALWAYS, preserve_metadata: bool = True) -> None:
        """
        Tries to save dataset to ``dataset_uri`` using all registered dataset savers.

        Parameters
        ----------
        dataset_uri:
            A URI to save to.
        compute_digest:
            Compute digest over the data when saving?
        preserve_metadata:
            When saving a dataset, store its metadata as well?
        """

        for saver in self.savers:
            if saver.can_save(dataset_uri):
                saver.save(self, dataset_uri, compute_digest=compute_digest, preserve_metadata=preserve_metadata)
                return

        raise exceptions.DatasetUriNotSupportedError("No known saver could save dataset to '{dataset_uri}'.".format(dataset_uri=dataset_uri))

    def is_lazy(self) -> bool:
        """
        Return whether this dataset instance is lazy and not all data has been loaded.

        Returns
        -------
        ``True`` if this dataset instance is lazy.
        """

        return self._load_lazy is not None

    def load_lazy(self) -> None:
        """
        Read all the data and convert the dataset to a non-lazy one.
        """

        if self._load_lazy is not None:
            self._load_lazy(self)

    # TODO: Allow one to specify priority which would then insert loader at a different place and not at the end?
    @classmethod
    def register_loader(cls, loader: Loader) -> None:
        """
        Registers a new dataset loader.

        Parameters
        ----------
        loader:
            An instance of the loader class implementing a new loader.
        """

        cls.loaders.append(loader)

    # TODO: Allow one to specify priority which would then insert saver at a different place and not at the end?
    @classmethod
    def register_saver(cls, saver: Saver) -> None:
        """
        Registers a new dataset saver.

        Parameters
        ----------
        saver:
            An instance of the saver class implementing a new saver.
        """

        cls.savers.append(saver)

    def __repr__(self) -> str:
        return self.__str__()

    def _get_description_keys(self) -> typing.Sequence[str]:
        return 'id', 'name', 'location_uris'

    def __str__(self) -> str:
        metadata = self.metadata.query(())

        return '{class_name}({description})'.format(
            class_name=type(self).__name__,
            description=', '.join('{key}=\'{value}\''.format(key=key, value=metadata[key]) for key in self._get_description_keys() if key in metadata),
        )

    def copy(self: D) -> D:
        # Metadata is copied from provided iterable.
        return type(self)(resources=self, load_lazy=self._load_lazy)

    def __copy__(self: D) -> D:
        return self.copy()

    def select_rows(self: D, row_indices_to_keep: typing.Mapping[str, typing.Sequence[int]]) -> D:
        """
        Generate a new Dataset from the row indices for DataFrames.

        Parameters
        ----------
        row_indices_to_keep:
            This is a dict where key is resource ID and value is a sequence of row indices to keep.
            If a resource ID is missing, the whole related resource is kept.

        Returns
        -------
        Returns a new Dataset.
        """

        resources = {}
        metadata = self.metadata

        for resource_id, resource in self.items():
            # We keep any resource which is missing from "row_indices_to_keep".
            if resource_id not in row_indices_to_keep:
                resources[resource_id] = resource
            else:
                if not isinstance(resource, container_pandas.DataFrame):
                    raise exceptions.InvalidArgumentTypeError("Only DataFrame resources can have rows selected, not '{type}'.".format(type=type(resource)))

                row_indices = sorted(row_indices_to_keep[resource_id])
                resources[resource_id] = self[resource_id].iloc[row_indices, :].reset_index(drop=True)

                # TODO: Expose this as a general metadata method.
                #       In that case this has to be done recursively over all nested ALL_ELEMENTS.
                #       Here we are operating at resource level so we have to iterate only over first
                #       ALL_ELEMENTS and resource's element itself.

                # Change the metadata. Update the number of rows in the split.
                # This makes a copy so that we can modify metadata in-place.
                metadata = metadata.update(
                    (resource_id,),
                    {
                        'dimension': {
                            'length': len(row_indices),
                        },
                    },
                )

                # Remove all rows not in this split and reorder those which are.
                for element_metadata_entry in [
                    metadata._current_metadata.all_elements,
                    metadata._current_metadata.elements[resource_id],
                ]:
                    if element_metadata_entry is None:
                        continue

                    elements = element_metadata_entry.elements
                    new_elements_evolver = utils.EMPTY_PMAP.evolver()
                    for i, row_index in enumerate(row_indices):
                        if row_index in elements:
                            new_elements_evolver.set(i, elements[row_index])
                    element_metadata_entry.elements = new_elements_evolver.persistent()
                    element_metadata_entry.is_elements_empty = not element_metadata_entry.elements
                    element_metadata_entry.update_is_empty()

        return type(self)(resources, metadata)

    def get_relations_graph(self) -> typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]]:
        """
        Builds the relations graph for the dataset.

        Each key in the output corresponds to a resource/table. The value under a key is the list of
        edges this table has. The edge is represented by a tuple of four elements. For example,
        if the edge is ``(resource_id, True, index_1, index_2, custom_state)``, it
        means that there is a foreign key that points to table ``resource_id``. Specifically,
        ``index_1`` column in the current table points to ``index_2`` column in the table ``resource_id``.

        ``custom_state`` is an empty dict when returned from this method, but allows users
        of this graph to store custom state there.

        Returns
        -------
        Dict[str, List[Tuple[str, bool, int, int, Dict]]]
            Returns the relation graph in adjacency representation.
        """

        graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]] = collections.defaultdict(list)

        for resource_id in self.keys():
            if not issubclass(self.metadata.query((resource_id,))['structural_type'], container_pandas.DataFrame):
                continue

            columns_length = self.metadata.query((resource_id, metadata_base.ALL_ELEMENTS,))['dimension']['length']
            for index in range(columns_length):
                column_metadata = self.metadata.query((resource_id, metadata_base.ALL_ELEMENTS, index))

                if 'foreign_key' not in column_metadata:
                    continue

                if column_metadata['foreign_key']['type'] != 'COLUMN':
                    continue

                foreign_resource_id = column_metadata['foreign_key']['resource_id']

                # "COLUMN" foreign keys should not point to non-DataFrame resources.
                assert isinstance(self[foreign_resource_id], container_pandas.DataFrame), type(self[foreign_resource_id])

                if 'column_index' in column_metadata['foreign_key']:
                    foreign_index = column_metadata['foreign_key']['column_index']
                elif 'column_name' in column_metadata['foreign_key']:
                    foreign_index = self.metadata.get_column_index_from_column_name(column_metadata['foreign_key']['column_name'], at=(foreign_resource_id,))
                else:
                    raise exceptions.UnexpectedValueError("Invalid foreign key: {foreign_key}".format(foreign_key=column_metadata['foreign_key']))

                # "True" and "False" implies forward and backward relationships, respectively.
                graph[resource_id].append((foreign_resource_id, True, index, foreign_index, {}))
                graph[foreign_resource_id].append((resource_id, False, foreign_index, index, {}))

        return graph

    def get_column_references_by_column_index(self) -> typing.Dict[str, typing.Dict[metadata_base.ColumnReference, typing.List[metadata_base.ColumnReference]]]:
        references: typing.Dict[str, typing.Dict[metadata_base.ColumnReference, typing.List[metadata_base.ColumnReference]]] = {
            'confidence_for': {},
            'rank_for': {},
            'boundary_for': {},
            'foreign_key': {},
        }

        for resource_id, resource in self.items():
            if not isinstance(resource, container_pandas.DataFrame):
                continue

            resource_references = self.metadata.get_column_references_by_column_index(resource_id, at=(resource_id,))

            references['confidence_for'].update(resource_references['confidence_for'])
            references['rank_for'].update(resource_references['rank_for'])
            references['boundary_for'].update(resource_references['boundary_for'])
            references['foreign_key'].update(resource_references['foreign_key'])

        return references

    @classmethod
    def _canonical_dataset_description(cls, dataset_description: typing.Dict, *, set_no_value: bool = False) -> typing.Dict:
        """
        Currently, this is just removing any local URIs the description might have.
        """

        # Making a copy.
        dataset_description = dict(dataset_description)

        utils.filter_local_location_uris(dataset_description, empty_value=metadata_base.NO_VALUE if set_no_value else None)

        return dataset_description

    def to_json_structure(self, *, canonical: bool = False) -> typing.Dict:
        """
        Returns only a top-level dataset description.
        """

        # Using "to_json_structure" and not "to_internal_json_structure" because
        # it is not indented that this would be parsed back directly, but just used
        # to know where to find the dataset.
        dataset_description = utils.to_json_structure(self.metadata.query(()))

        if canonical:
            dataset_description = self._canonical_dataset_description(dataset_description)

        metadata_base.CONTAINER_SCHEMA_VALIDATOR.validate(dataset_description)

        return dataset_description

    @classmethod
    def _canonical_metadata(cls, metadata: metadata_base.DataMetadata) -> metadata_base.DataMetadata:
        """
        Currently, this is just removing any local URIs the metadata might have.
        """

        metadata = metadata.update((), cls._canonical_dataset_description(metadata.query(()), set_no_value=True))

        metadata = cls._canonical_metadata_traverse(metadata, metadata, [])

        return metadata

    @classmethod
    def _canonical_metadata_traverse(cls, metadata: metadata_base.DataMetadata, output_metadata: metadata_base.DataMetadata, selector: metadata_base.ListSelector) -> metadata_base.DataMetadata:
        # "ALL_ELEMENTS" is always first, if it exists, which works in our favor here.
        elements = metadata.get_elements(selector)

        for element in elements:
            new_selector = selector + [element]
            new_metadata = dict(metadata._query(new_selector, metadata._current_metadata, 0))
            utils.filter_local_location_uris(new_metadata, empty_value=metadata_base.NO_VALUE)
            output_metadata = output_metadata.update(new_selector, new_metadata)

            output_metadata = cls._canonical_metadata_traverse(metadata, output_metadata, new_selector)

        return output_metadata


def dataset_serializer(obj: Dataset) -> dict:
    data = {
        'metadata': obj.metadata,
        'dataset': dict(obj),
    }

    if type(obj) is not Dataset:
        data['type'] = type(obj)

    return data


def dataset_deserializer(data: dict) -> Dataset:
    dataset = data.get('type', Dataset)(data['dataset'], data['metadata'])
    return dataset


if pyarrow_lib is not None:
    pyarrow_lib._default_serialization_context.register_type(
        Dataset, 'd3m.dataset',
        custom_serializer=dataset_serializer,
        custom_deserializer=dataset_deserializer,
    )


def get_dataset(
    dataset_uri: str, *, compute_digest: ComputeDigest = ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False, lazy: bool = False,
    datasets_dir: str = None, handle_score_split: bool = True,
) -> Dataset:
    if datasets_dir is not None:
        datasets, problem_descriptions = utils.get_datasets_and_problems(datasets_dir, handle_score_split)

        if dataset_uri in datasets:
            dataset_uri = datasets[dataset_uri]

    dataset_uri = utils.fix_uri(dataset_uri)

    return Dataset.load(dataset_uri, compute_digest=compute_digest, strict_digest=strict_digest, lazy=lazy)


def describe_handler(arguments: argparse.Namespace, *, dataset_resolver: typing.Callable = None) -> None:
    if dataset_resolver is None:
        dataset_resolver = get_dataset

    output_stream = getattr(arguments, 'output', sys.stdout)

    has_errored = False

    for dataset_path in arguments.datasets:
        if getattr(arguments, 'list', False):
            print(dataset_path, file=output_stream)

        try:
            start_timestamp = time.perf_counter()
            dataset = dataset_resolver(
                dataset_path,
                compute_digest=ComputeDigest[getattr(arguments, 'compute_digest', ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
                lazy=getattr(arguments, 'lazy', False),
            )
            end_timestamp = time.perf_counter()
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error loading dataset: {dataset_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error loading dataset: {dataset_path}") from error

        try:
            if getattr(arguments, 'print', False) or getattr(arguments, 'metadata', False) or getattr(arguments, 'time', False):
                if getattr(arguments, 'print', False):
                    pprint.pprint(dataset, stream=output_stream)
                if getattr(arguments, 'metadata', False):
                    dataset.metadata.pretty_print(handle=output_stream)
                if getattr(arguments, 'time', False):
                    print(f"Time: {(end_timestamp - start_timestamp):.3f}s", file=output_stream)
            else:
                dataset_description = dataset.to_json_structure(canonical=True)

                json.dump(
                    dataset_description,
                    output_stream,
                    indent=(getattr(arguments, 'indent', 2) or None),
                    sort_keys=getattr(arguments, 'sort_keys', False),
                    allow_nan=False,
                )  # type: ignore
                output_stream.write('\n')
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error describing dataset: {dataset_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error describing dataset: {dataset_path}") from error

    if has_errored:
        sys.exit(1)


def convert_handler(arguments: argparse.Namespace, *, dataset_resolver: typing.Callable = None) -> None:
    if dataset_resolver is None:
        dataset_resolver = get_dataset

    try:
        dataset = dataset_resolver(
            arguments.input_uri,
            compute_digest=ComputeDigest[getattr(arguments, 'compute_digest', ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
    except Exception as error:
        raise Exception(f"Error loading dataset '{arguments.input_uri}'.") from error

    output_uri = utils.fix_uri(arguments.output_uri)

    try:
        dataset.save(output_uri, preserve_metadata=getattr(arguments, 'preserve_metadata', True))
    except Exception as error:
        raise Exception(f"Error saving dataset '{arguments.input_uri}' to '{output_uri}'.") from error


def main(argv: typing.Sequence) -> None:
    raise exceptions.NotSupportedError("This CLI has been removed. Use \"python3 -m d3m dataset describe\" instead.")


if __name__ == '__main__':
    main(sys.argv)
