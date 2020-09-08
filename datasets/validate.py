#!/usr/bin/env python3
#
# This script validates that problem and dataset descriptions match
# standards and conventions (schemas, naming and directory structure, etc.).
#
# This script expects a that there is a clone of the "data-supply"
# repository in the same directory as this script.
#
# Checks done by this script:
#  - Dataset description validates according to its schema.
#  - Problem description validates according to its schema.
#  - Dataset description filename should be "datasetDoc.json".
#  - Problem description filename should be "problemDoc.json".
#  - There should be no duplicate dataset IDs or problem IDs.
#  - Dataset directory names should match the dataset IDs, and be under
#    a matching parent directory based on that ID (where ID should
#    have an expected suffix).
#  - All problem descriptions for dataset views/splits should be the same.
#  - Dataset splits should match in ID the original dataset based on the directory
#    structure they are in, but have "TEST, "TRAIN", or "SCORE" suffix.
#  - Problem descriptions should reference existing datasets and columns.
#  - Dataset and problem descriptions should be (almost) equal between splits.
#  - Clustering problems require numClusters in target specifications.
#  - Clustering problems should not have data splitting configuration.
#  - Test and train split of datasets used in clustering problems should be the same.
#  - Require dataset digest.
#  - Dataset entry points should have "learningData" as resource ID.
#  - Problem descriptions using "f1", "precision", "recall", and "jaccardSimilarityScore"
#    metrics should have only two distinct values in target columns, have "posLabel" provided,
#    and that "posLabel" value should be among target values.
#  - No other should have "posLabel" set.
#  - "hammingLoss" metric can be used only with multi-label problems.
#  - "precisionAtTopK" should be used only with forecasting.
#  - Problem descriptions should have only one target, except for multi-variate
#    and object detection problems which should have more than one.
#  - Dataset entry point cannot be a collection.
#  - Dataset entry point has to have columns metadata.
#  - There is at most one "index" or "multiIndex" column per resource.
#  - "index" and "multiIndex" cannot be set at the same time.
#  - Dataset entry point is required to have an "index" or "multiIndex" column.
#  - Columns cannot be both "index" and "key" at the same time.
#  - Columns cannot be both "multiIndex" and "key" at the same time.
#  - "index" columns have to have unique values and no missing values.
#  - "multiIndex" columns have to have no missing values.
#  - "key" columns have to have unique values.
#  - Every metric should be listed only once in a problem description.
#  - Some task keywords can be used only with corresponding task keywords.
#  - All resource formats used by a resource should be from the standard list of them.
#  - All files used in a collection resource should have a file extension of a resource
#    format from the standard list of them.
#  - Collection resource should contain at least one file.
#  - Resource path of a collection resource should end with "/".
#  - Any file referenced in a collection resource must exist.
#  - On edgelist resources, both "edgeSource" and "edgeTarget" columns should exist in
#    same resource, only one each. It should have additional two column roles for direction
#    and simple/multi. Those should match between columns (so both should be directed or not,
#    and simple or multi, but not mix).
#  - When there is "multiIndex" column, all rows for same index value should have the same
#    values in all columns except "suggestedTarget" columns.
#  - Makes sure that "columnsCount" matches the number of columns, when it exists.

import argparse
import collections
import copy
import functools
import json
import traceback
import os
import os.path
import sys

import cerberus
import deep_dircmp
import pandas

LIMIT_OUTPUT = 10
EDGELIST_COLUMN_ROLES = [
    'edgeSource',
    'directedEdgeSource',
    'undirectedEdgeSource',
    'multiEdgeSource',
    'simpleEdgeSource',
    'edgeTarget',
    'directedEdgeTarget',
    'undirectedEdgeTarget',
    'multiEdgeTarget',
    'simpleEdgeTarget',
]

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data-supply')):
    raise Exception("\"data-supply\" directory is missing. You should clone the repository to be in the same directory as this script.")

with open(os.path.join(os.path.dirname(__file__), 'data-supply', 'schemas', 'datasetSchema.json')) as dataset_description_schema_file:
    dataset_description_validator = cerberus.Validator(json.load(dataset_description_schema_file))

with open(os.path.join(os.path.dirname(__file__), 'data-supply', 'schemas', 'problemSchema.json')) as problem_description_schema_file:
    problem_description_validator = cerberus.Validator(json.load(problem_description_schema_file))

with open(os.path.join(os.path.dirname(__file__), 'data-supply', 'documentation', 'supportedResourceTypesFormats.json')) as supported_resource_types_formats_file:
    supported_resource_types_formats = json.load(supported_resource_types_formats_file)
    res_format_to_extensions = {}
    for supported_resource in supported_resource_types_formats['supported_resource_types_and_formats']:
        for res_format, extensions in supported_resource['resFormat'].items():
            if res_format not in res_format_to_extensions:
                res_format_to_extensions[res_format] = sorted(set(extensions))
            else:
                res_format_to_extensions[res_format] = sorted(set(extensions) | set(res_format_to_extensions[res_format]))


@functools.lru_cache(maxsize=10)
def read_csv(data_path):
    return pandas.read_csv(
        data_path,
        # We do not want to do any conversion of values.
        dtype=str,
        # We always expect one row header.
        header=0,
        # We want empty strings and not NaNs.
        na_filter=False,
        encoding='utf8',
    )


def validate_dataset_path(description_id, description_path, *, strict_naming=True):
    if os.path.basename(description_path) != 'datasetDoc.json':
        print("ERROR: Dataset description filename is not 'datasetDoc.json'.")
        return True

    if strict_naming:
        split_path = os.path.dirname(description_path).split(os.sep)
        for suffix in ['_dataset_TEST', '_dataset_TRAIN', '_dataset_SCORE']:
            if description_id.endswith(suffix):
                expected_paths = [[description_id[:-len(suffix)], suffix[len('_dataset_'):], suffix[1:]]]

                # A special case, SCORE dataset/problem can be in TEST directory.
                if suffix == '_dataset_SCORE':
                    expected_paths.append([description_id[:-len(suffix)], suffix[len('_dataset_'):], 'dataset_TEST'])

                if split_path[-3:] not in expected_paths:
                    print("ERROR: Dataset directory path {directory_path} does not match any of expected paths: {expected_paths}".format(
                        directory_path=split_path[-3:],
                        expected_paths=', '.join(str(expected_path) for expected_path in expected_paths),
                    ))
                    return True

                break
        else:
            if not description_id.endswith('_dataset'):
                print("ERROR: Dataset ID does not end with allowed suffix: {description_id}".format(
                    description_id=description_id,
                ))
                return True

            expected_path = [description_id[:-len('_dataset')], description_id]

            if split_path[-2:] != expected_path:
                print("ERROR: Dataset directory path {directory_path} does not match expected path: {expected_path}".format(
                    directory_path=split_path[-2:],
                    expected_path=expected_path,
                ))
                return True

    return False


def validate_metrics(problem_description):
    error = False

    existing_metrics = set()
    for metric in problem_description.get('inputs', {}).get('performanceMetrics', []):
        if metric['metric'] in ['f1', 'precision', 'recall', 'jaccardSimilarityScore']:
            if 'posLabel' not in metric:
                print("ERROR: Problem uses '{metric}' metric, but 'posLabel' is not provided.".format(
                    metric=metric['metric'],
                ))
                error = True
            if set(problem_description['about']['taskKeywords']) & {'multiClass', 'multiLabel'}:
                print("ERROR: Problem uses '{metric}' metric, but it is a multi-class or a multi-label problem.".format(
                    metric=metric['metric'],
                ))
                error = True
        elif 'posLabel' in metric:
            print("ERROR: Problem does not use 'f1', 'precision', 'recall', or 'jaccardSimilarityScore' metric, but 'posLabel' is provided.".format(
                metric=metric['metric'],
            ))
            error = True

        if metric['metric'] == 'hammingLoss' and 'multiLabel' not in set(problem_description['about']['taskKeywords']):
            print("ERROR: Problem uses 'hammingLoss' metric, but it is not a multi-label problem.")
            error = True

        if metric['metric'] == 'precisionAtTopK' and 'forecasting' not in set(problem_description['about']['taskKeywords']):
            print("ERROR: Problem uses 'precisionAtTopK' metric, but it is not forecasting problem.")
            error = True

        if metric['metric'] in existing_metrics:
            print("ERROR: Problem uses same metric '{metric}' multiple times.".format(metric=metric['metric']))
            error = True
        existing_metrics.add(metric['metric'])

    return error


def validate_keywords(problem_description):
    task_keywords = set(problem_description['about']['taskKeywords'])

    targets_number = 0
    for data in problem_description.get('inputs', {}).get('data', []):
        targets_number += len(data.get('targets', []))

    if 'regression' in task_keywords and 'multivariate' in task_keywords:
        if targets_number < 2:
            print("ERROR: Problem is a multi-variate problem, but it does not have more than 1 target.")
            return True
    elif 'objectDetection' in task_keywords:
        if targets_number != 1 and targets_number != 2:
            print("ERROR: Problem is an object detection problem, but it does not have 1 or 2 targets.")
            return True
    elif targets_number != 1:
        print("ERROR: Problem has more than 1 target.")
        return True

    if task_keywords & {'binary', 'multiClass', 'multiLabel'} and not task_keywords & {'classification', 'vertexClassification'}:
        print("ERROR: Invalid combination of problem's keywords: {task_keywords}".format(
            task_keywords=task_keywords,
        ))
        return True
    if task_keywords & {'classification', 'vertexClassification'} and not task_keywords & {'binary', 'multiClass', 'multiLabel'}:
        print("ERROR: Invalid combination of problem's keywords: {task_keywords}".format(
            task_keywords=task_keywords,
        ))
        return True

    if task_keywords & {'univariate', 'multivariate'} and 'regression' not in task_keywords:
        print("ERROR: Invalid combination of problem's keywords: {task_keywords}".format(
            task_keywords=task_keywords,
        ))
        return True
    if 'regression' in task_keywords and not task_keywords & {'univariate', 'multivariate'}:
        print("ERROR: Invalid combination of problem's keywords: {task_keywords}".format(
            task_keywords=task_keywords,
        ))
        return True

    if task_keywords & {'overlapping', 'nonOverlapping'} and not task_keywords & {'clustering', 'communityDetection'}:
        print("ERROR: Invalid combination of problem's keywords: {task_keywords}".format(
            task_keywords=task_keywords,
        ))
        return True
    if task_keywords & {'clustering', 'communityDetection'} and not task_keywords & {'overlapping', 'nonOverlapping'}:
        print("ERROR: Invalid combination of problem's keywords: {task_keywords}".format(
            task_keywords=task_keywords,
        ))
        return True

    return False


def validate_files(dataset_description_path, data_resource, dataset_description, column_index, collection_resource_id):
    for collection_data_resource in dataset_description['dataResources']:
        if collection_data_resource['resID'] == collection_resource_id:
            break
    else:
        print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' referencing with column {column_index} a collection resource '{collection_resource_id}', but the resource does not exixt.".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
            column_index=column_index,
            collection_resource_id=collection_resource_id,
        ))
        # We cannot do much more here.
        return True

    if not collection_data_resource.get('isCollection', False):
        print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' referencing with column {column_index} a collection resource '{collection_resource_id}', but the resource is not a collection.".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
            column_index=column_index,
            collection_resource_id=collection_resource_id,
        ))
        # We cannot do much more here.
        return True

    error = False

    data_path = os.path.join(os.path.dirname(dataset_description_path), data_resource['resPath'])

    data = read_csv(data_path)

    collection_dir = os.path.join(os.path.dirname(dataset_description_path), collection_data_resource['resPath'])

    count = 0
    for filename in data.iloc[:, column_index]:
        filepath = os.path.join(collection_dir, filename)

        if not os.path.isfile(filepath):
            count += 1

            print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' referencing with column {column_index} a file in a collection resource '{collection_resource_id}', but the file does not exist: {filename}".format(
                dataset_path=dataset_description_path,
                resource_id=data_resource['resID'],
                column_index=column_index,
                collection_resource_id=collection_resource_id,
                filename=filename,
            ))
            error = True

        if LIMIT_OUTPUT is not None and count > LIMIT_OUTPUT:
            break

    return error


def validate_collection(dataset_description_path, data_resource):
    error = False

    if not data_resource['resPath'].endswith('/'):
        print("ERROR: Dataset '{dataset_path}' has a collection resource '{resource_id}' where resource path is not ending with '/': {res_path}".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
            res_path=data_resource['resPath'],
        ))
        error = True

    allowed_file_extensions = set()
    for res_format, extensions in data_resource['resFormat'].items():
        unsupported_extensions = set(extensions) - set(res_format_to_extensions[res_format])
        if unsupported_extensions:
            print("ERROR: Dataset '{dataset_path}' has a collection resource '{resource_id}' and resource format '{res_format}' with unsupported extensions: {unsupported_extensions}".format(
                dataset_path=dataset_description_path,
                resource_id=data_resource['resID'],
                res_format=res_format,
                unsupported_extensions=sorted(unsupported_extensions),
            ))
            error = True
        allowed_file_extensions.update(extensions)

    collection_dir = os.path.join(os.path.dirname(dataset_description_path), data_resource['resPath'])
    is_empty = True
    count = 0
    for dirpath, dirnames, filenames in os.walk(collection_dir):
        for filename in filenames:
            is_empty = False

            filepath = os.path.join(dirpath, filename)

            file_extension = get_file_extension(filepath)
            if file_extension not in allowed_file_extensions:
                count += 1

                print("ERROR: Dataset '{dataset_path}' has a collection resource '{resource_id}' with a file with unsupported file extension: {filepath}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    filepath=filepath,
                ))
                error = True

            if LIMIT_OUTPUT is not None and count > LIMIT_OUTPUT:
                break

        if LIMIT_OUTPUT is not None and count > LIMIT_OUTPUT:
            break

    if is_empty:
        print("ERROR: Dataset '{dataset_path}' has a collection resource '{resource_id}' without any files.".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
        ))
        error = True

    return error


def validate_multi_index(dataset_description_path, data_resource, multi_index_column):
    error = False

    suggested_target_columns = []
    for column_description in data_resource['columns']:
        if 'suggestedTarget' in column_description['role']:
            suggested_target_columns.append(column_description['colIndex'])

    data_path = os.path.join(os.path.dirname(dataset_description_path), data_resource['resPath'])

    data = read_csv(data_path)

    attribute_columns = [column_index for column_index in range(len(data.columns)) if column_index != multi_index_column and column_index not in suggested_target_columns]
    attributes = data.iloc[:, attribute_columns].set_index(data.iloc[:, multi_index_column])

    count = 0
    for group_name, group in attributes.groupby(level=0):
        # The first row in a group is not marked, so we add 1 to number of duplicated rows.
        if group.duplicated(keep='first').sum() + 1 != len(group):
            count += 1

            print("ERROR: Dataset '{dataset_path}' has a multi-index resource '{resource_id}' with all attributes in rows not equal for index value '{value}'.".format(
                dataset_path=dataset_description_path,
                resource_id=data_resource['resID'],
                value=group_name,
            ))
            error = True

        if LIMIT_OUTPUT is not None and count > LIMIT_OUTPUT:
            break

    return error


def validate_edgelist(dataset_description_path, data_resource):
    error = False

    found_source = False
    is_directed_source = None
    is_multi_source = None
    found_target = False
    is_directed_target = None
    is_multi_target = None
    for column_description in data_resource['columns']:
        if 'edgeSource' in column_description['role']:
            # We have to check this only here or only in "edgeTarget" case. We check it here.
            if 'edgeTarget' in column_description['role']:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting source vs. target column roles.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True

            if found_source:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with multiple edge source columns.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True
                continue
            found_source = True

            if 'multiEdgeSource' in column_description['role']:
                if is_multi_source is None:
                    is_multi_source = True
                elif is_multi_source != True:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting multi vs. simple column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if 'simpleEdgeSource' in column_description['role']:
                if is_multi_source is None:
                    is_multi_source = False
                elif is_multi_source != False:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting multi vs. simple column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if is_multi_source is None:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with missing multi vs. simple column role.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True

            if 'directedEdgeSource' in column_description['role']:
                if is_directed_source is None:
                    is_directed_source = True
                elif is_directed_source != True:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting directed vs. undirected column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if 'undirectedEdgeSource' in column_description['role']:
                if is_directed_source is None:
                    is_directed_source = False
                elif is_directed_source != False:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting directed vs. undirected column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if is_directed_source is None:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with missing directed vs. undirected column role.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True

        if 'edgeTarget' in column_description['role']:
            if found_target:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with multiple edge target columns.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True
                continue
            found_target = True

            if 'multiEdgeTarget' in column_description['role']:
                if is_multi_target is None:
                    is_multi_target = True
                elif is_multi_target != True:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting multi vs. simple column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if 'simpleEdgeTarget' in column_description['role']:
                if is_multi_target is None:
                    is_multi_target = False
                elif is_multi_target != False:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting multi vs. simple column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if is_multi_target is None:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with missing multi vs. simple column role.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True

            if 'directedEdgeTarget' in column_description['role']:
                if is_directed_target is None:
                    is_directed_target = True
                elif is_directed_target != True:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting directed vs. undirected column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if 'undirectedEdgeTarget' in column_description['role']:
                if is_directed_target is None:
                    is_directed_target = False
                elif is_directed_target != False:
                    print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting directed vs. undirected column roles.".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                    ))
                    error = True

            if is_directed_target is None:
                print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with missing directed vs. undirected column role.".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                ))
                error = True

    if not found_source:
        print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with missing edge source column role.".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
        ))
        error = True
    if not found_target:
        print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with missing edge target column role.".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
        ))
        error = True

    if found_source and found_target:
        if is_directed_source != is_directed_target:
            print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting directed vs. undirected column roles.".format(
                dataset_path=dataset_description_path,
                resource_id=data_resource['resID'],
            ))
            error = True

        if is_multi_source != is_multi_target:
            print("ERROR: Dataset '{dataset_path}' has a edgelist resource '{resource_id}' with conflicting multi vs. simple column roles.".format(
                dataset_path=dataset_description_path,
                resource_id=data_resource['resID'],
            ))
            error = True

    return error


def get_file_extension(path):
    extension = os.path.splitext(path)[1]
    if extension:
        # We remove leading dot as returned from "splitext".
        return extension[1:]
    else:
        raise ValueError(f"Cannot get file extension of '{path}'.")


def validate_dataset(dataset_description_path, dataset_description):
    error = False

    for data_resource in dataset_description['dataResources']:
        if os.path.splitext(os.path.basename(data_resource['resPath']))[0] == 'learningData' and data_resource['resID'] != 'learningData':
            print("ERROR: Dataset '{dataset_path}' has a dataset entry point without 'learningData' as resource's ID, but '{resource_id}'.".format(
                dataset_path=dataset_description_path,
                resource_id=data_resource['resID'],
            ))
            error = True

        if data_resource['resID'] == 'learningData':
            if data_resource.get('isCollection', False):
                print("ERROR: Dataset '{dataset_path}' has a dataset entry point which is a collection.".format(
                    dataset_path=dataset_description_path,
                ))
                error = True

            if 'columns' not in data_resource:
                print("ERROR: Dataset '{dataset_path}' has a dataset entry point without columns metadata.".format(
                    dataset_path=dataset_description_path,
                ))
                error = True

        if 'columns' in data_resource:
            index_columns = []
            multi_index_columns = []
            key_columns = []
            edgelist_columns = []
            for column_description in data_resource['columns']:
                if 'index' in column_description['role']:
                    index_columns.append(column_description['colIndex'])
                if 'multiIndex' in column_description['role']:
                    multi_index_columns.append(column_description['colIndex'])
                if 'key' in column_description['role']:
                    key_columns.append(column_description['colIndex'])
                if any(edgelist_column_role in column_description['role'] for edgelist_column_role in EDGELIST_COLUMN_ROLES):
                    edgelist_columns.append(column_description['colIndex'])

            index_columns_set = set(index_columns)
            multi_index_columns_set = set(multi_index_columns)
            key_columns_set = set(key_columns)

            if index_columns_set & multi_index_columns_set:
                print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with columns being both index and multi-index at the same time: {index_columns}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    index_columns=sorted(index_columns_set & multi_index_columns_set),
                ))
                error = True
            elif data_resource['resID'] == 'learningData' and len(index_columns) + len(multi_index_columns) == 0:
                print("ERROR: Dataset '{dataset_path}' has a dataset entry point with no index columns.".format(
                    dataset_path=dataset_description_path,
                ))
                error = True
            elif len(index_columns) + len(multi_index_columns) > 1:
                print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with multiple index columns: {index_columns}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    index_columns=index_columns + multi_index_columns,
                ))
                error = True

            if index_columns_set & key_columns_set:
                print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with columns being both index and key at the same time: {index_columns}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    index_columns=sorted(index_columns_set & key_columns_set),
                ))
                error = True

            if multi_index_columns_set & key_columns_set:
                print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with columns being both multi-index and key at the same time: {index_columns}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    index_columns=sorted(multi_index_columns_set & key_columns_set),
                ))
                error = True

            if data_resource.get('isCollection', False):
                continue

            for column_index in index_columns:
                error = validate_column_values(dataset_description_path, data_resource, column_index, unique=True, no_missing=True) or error
            for column_index in multi_index_columns:
                error = validate_column_values(dataset_description_path, data_resource, column_index, unique=False, no_missing=True) or error
            for column_index in key_columns:
                error = validate_column_values(dataset_description_path, data_resource, column_index, unique=True, no_missing=False) or error

            for column_description in data_resource['columns']:
                if 'refersTo' in column_description and column_description['refersTo']['resObject'] == 'item':
                    error = validate_files(dataset_description_path, data_resource, dataset_description, column_description['colIndex'], column_description['refersTo']['resID']) or error

            if edgelist_columns:
                error = validate_edgelist(dataset_description_path, data_resource) or error

            if len(multi_index_columns) == 1:
                error = validate_multi_index(dataset_description_path, data_resource, multi_index_columns[0]) or error

        for res_format in data_resource['resFormat'].keys():
            if res_format not in res_format_to_extensions:
                print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with unsupported format: {res_format}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    res_format=res_format,
                ))
                error = True

        if data_resource.get('isCollection', False):
            error = validate_collection(dataset_description_path, data_resource) or error
        else:
            if len(data_resource['resFormat']) == 1:
                file_extension = get_file_extension(data_resource['resPath'])
                # There should be only one resource format listed for non-collection resources.
                if file_extension not in list(data_resource['resFormat'].values())[0]:
                    print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with invalid resource path file extension: {file_extension}".format(
                        dataset_path=dataset_description_path,
                        resource_id=data_resource['resID'],
                        file_extension=file_extension,
                    ))
                    error = True
            else:
                print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with invalid number of listed formats: {count}".format(
                    dataset_path=dataset_description_path,
                    resource_id=data_resource['resID'],
                    count=len(data_resource['resFormat']),
                ))
                error = True

    return error


def validate_dataset_description(dataset_description_path, known_dataset_descriptions, *, strict_naming=True):
    print("Validating dataset '{dataset_description_path}'.".format(dataset_description_path=dataset_description_path))

    try:
        with open(dataset_description_path) as dataset_description_file:
            dataset_description = json.load(dataset_description_file)

        if not dataset_description_validator.validate(dataset_description):
            print("ERROR: Schema validation: {errors}".format(errors=dataset_description_validator.errors))
            return True

        dataset_id = dataset_description['about']['datasetID']

        # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
        # They are the same as TEST dataset splits, but we present them differently, so that
        # SCORE dataset splits have targets as part of data. Because of this we also update
        # corresponding dataset ID.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
        if os.path.exists(os.path.join(os.path.dirname(dataset_description_path), '..', 'targets.csv')) and dataset_id.endswith('_TEST'):
            dataset_id = dataset_id[:-5] + '_SCORE'
        if dataset_id in known_dataset_descriptions:
            print("ERROR: Duplicate dataset ID '{dataset_id}': '{first_path}' and '{second_path}'".format(
                dataset_id=dataset_id,
                first_path=known_dataset_descriptions[dataset_id]['path'],
                second_path=dataset_description_path,
            ))
            return True

        known_dataset_descriptions[dataset_id] = {
            'path': dataset_description_path,
            'description': dataset_description,
        }

        if validate_dataset_path(dataset_id, dataset_description_path, strict_naming=strict_naming):
            return True

        #if 'digest' not in dataset_description['about']:
        #    print("ERROR: Dataset '{dataset_path}' missing digest.".format(dataset_path=dataset_description_path))
        #    return True

        if validate_dataset(dataset_description_path, dataset_description):
            return True

    except Exception:
        print("ERROR: Unexpected exception:")
        traceback.print_exc()
        return True

    return False


def validate_problem_description(problem_description_path, known_problem_descriptions):
    print("Validating problem '{problem_description_path}'.".format(problem_description_path=problem_description_path))

    try:
        with open(problem_description_path) as problem_description_file:
            problem_description = json.load(problem_description_file)

        if not problem_description_validator.validate(problem_description):
            print("ERROR: Schema validation: {errors}".format(errors=problem_description_validator.errors))
            return True

        problem_id = problem_description['about']['problemID']

        # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
        # They are the same as TEST dataset splits, but we present them differently, so that
        # SCORE dataset splits have targets as part of data. Because of this we also update
        # corresponding problem ID.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
        if os.path.exists(os.path.join(os.path.dirname(problem_description_path), '..', 'targets.csv')) and problem_id.endswith('_TEST'):
            problem_id = problem_id[:-5] + '_SCORE'

            # Also update dataset references.
            for data in problem_description.get('inputs', {}).get('data', []):
                if data['datasetID'].endswith('_TEST'):
                    data['datasetID'] = data['datasetID'][:-5] + '_SCORE'

        # All problem descriptions show be the same.
        if problem_id.endswith('_TRAIN') or problem_id.endswith('_TEST') or problem_id.endswith('_SCORE'):
            print("ERROR: Invalid problem ID '{problem_id}' in '{problem_description_path}'.".format(
                problem_id=problem_id,
                problem_description_path=problem_description_path,
            ))
            return True

        if problem_id in known_problem_descriptions:
            # Problem descriptions with same ID should have the same content.
            if problem_description == known_problem_descriptions[problem_id]['description']:
                known_problem_descriptions[problem_id]['paths'].append(problem_description_path)
            else:
                print("ERROR: Duplicate problem ID '{problem_id}', but different problem description: {first_paths} and '{second_path}'".format(
                    problem_id=problem_id,
                    first_paths=known_problem_descriptions[problem_id]['paths'],
                    second_path=problem_description_path,
                ))
                return True

        else:
            known_problem_descriptions[problem_id] = {
                'paths': [problem_description_path],
                'description': problem_description,
            }

        if os.path.basename(problem_description_path) != 'problemDoc.json':
            print("ERROR: Problem description filename '{problem_description_path}' is not 'problemDoc.json'.".format(
                problem_description_path=problem_description_path,
            ))
            return True

        if validate_metrics(problem_description):
            return True

        if validate_keywords(problem_description):
            return True

        split_path = os.path.dirname(problem_description_path).split(os.sep)
        for split_directory in ['problem_TRAIN', 'problem_TEST', 'problem_SCORE']:
            if split_directory in split_path and 'datasetViewMaps' not in problem_description.get('inputs', {}).get('dataSplits', {}):
                print("ERROR: Problem '{problem_description_path}' is missing dataset view maps.".format(
                    problem_description_path=problem_description_path,
                ))
                return True

    except Exception:
        print("ERROR: Unexpected exception:")
        traceback.print_exc()
        return True

    return False


def validate_column_values(dataset_description_path, data_resource, column_index, *, unique, no_missing):
    error = False

    data_path = os.path.join(os.path.dirname(dataset_description_path), data_resource['resPath'])

    data = read_csv(data_path)

    column_values = data.iloc[:, column_index]

    # We assume missing values is represented as empty strings.
    column_values_without_missing = column_values[column_values != '']

    # There should be no NA anyway anymore.
    value_counts = column_values_without_missing.value_counts(dropna=True)

    if unique and (value_counts > 1).sum():
        duplicate = list(value_counts[value_counts > 1].keys())
        if LIMIT_OUTPUT is not None:
            duplicate = duplicate[:LIMIT_OUTPUT]

        print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with column {column_index} which should have unique values but it does not. Example duplicate values: {duplicate}".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
            column_index=column_index,
            duplicate=duplicate,
        ))
        error = True

    if no_missing and len(column_values) != len(column_values_without_missing):
        print("ERROR: Dataset '{dataset_path}' has a resource '{resource_id}' with column {column_index} which should have no missing values but it does have them.".format(
            dataset_path=dataset_description_path,
            resource_id=data_resource['resID'],
            column_index=column_index,
        ))
        error = True

    return error


def validate_target_values(problem_paths, dataset_path, problem_description, data_resource, target):
    error = False

    data_path = os.path.join(os.path.dirname(dataset_path), data_resource['resPath'])

    data = read_csv(data_path)

    target_values = data.iloc[:, target['colIndex']]
    distinct_values = list(target_values.value_counts(dropna=False).keys())
    number_distinct_values = len(distinct_values)
    # We assume missing values is represented as empty strings.
    has_missing_values = '' in distinct_values
    if has_missing_values:
        # We do not count missing values as distinct values.
        number_distinct_values -= 1
    task_keywords = set(problem_description['about']['taskKeywords'])

    if 'binary' in task_keywords:
        if number_distinct_values != 2:
            print("ERROR: Problem {problem_paths} has 'binary' keyword, but target column does not have 2 distinct values, but {number_distinct_values}.".format(
                problem_paths=problem_paths,
                number_distinct_values=number_distinct_values,
            ))
            error = True
    elif 'multiClass' in task_keywords:
        if number_distinct_values < 3:
            print("ERROR: Problem {problem_paths} has 'multiClass' keyword, but target column does not have more than 2 distinct values, but {number_distinct_values}.".format(
                problem_paths=problem_paths,
                number_distinct_values=number_distinct_values,
            ))
            error = True

    for metric in problem_description.get('inputs', {}).get('performanceMetrics', []):
        if metric['metric'] in ['f1', 'precision', 'recall', 'jaccardSimilarityScore']:
            if number_distinct_values != 2:
                print("ERROR: Problem {problem_paths} uses '{metric}' metric, but target column does not have 2 distinct values, but {number_distinct_values}.".format(
                    problem_paths=problem_paths,
                    metric=metric['metric'],
                    number_distinct_values=number_distinct_values,
                ))
                error = True
            if 'posLabel' in metric and metric['posLabel'] not in distinct_values:
                print("ERROR: Problem {problem_paths} provides 'posLabel' for metric '{metric}' with value '{value}', but possible values are: {distinct_values}".format(
                    problem_paths=problem_paths,
                    metric=metric['metric'],
                    value=metric['posLabel'],
                    distinct_values=sorted(distinct_values),
                ))
                error = True

    if has_missing_values and not task_keywords & {'semiSupervised', 'clustering'}:
        print("ERROR: Problem {problem_paths} has target column with missing values, but it not a semi-supervised or clustering task.".format(
            problem_paths=problem_paths,
        ))
        error = True
    if 'semiSupervised' in task_keywords and not has_missing_values:
        print("ERROR: Problem {problem_paths} is a semi-supervised task, but does not have a target column with missing values.".format(
            problem_paths=problem_paths,
        ))
        error = True

    return error


def get_all_columns(dataset_path, resource_id, data_resource):
    data_path = os.path.join(os.path.dirname(dataset_path), data_resource['resPath'])

    data = read_csv(data_path)

    data_columns = [{
        'colIndex': column_index,
        'colName': column_name,
        'colType': 'unknown',
        'role': []
    } for column_index, column_name in enumerate(data.columns)]

    columns = data_resource.get('columns', None)

    if columns is None:
        return data_columns

    if 'columnsCount' in data_resource and data_resource['columnsCount'] != len(data_columns):
        raise ValueError("Dataset '{dataset_path}' has resource '{resource_id}' with incorrect columns count {columns_count} (correct {correct_count}).".format(
            dataset_path=dataset_path,
            resource_id=resource_id,
            columns_count=data_resource['columnsCount'],
            correct_count=len(data_columns),
        ))

    if len(columns) >= len(data_columns):
        columns_names = [{'colIndex': c['colIndex'], 'colName': c['colName']} for c in columns]
        data_columns_names = [{'colIndex': c['colIndex'], 'colName': c['colName']} for c in data_columns]

        if columns_names != data_columns_names:
            raise ValueError("Dataset '{dataset_path}' has resource '{resource_id}' where metadata columns do not match data columns.".format(
                dataset_path=dataset_path,
                resource_id=resource_id,
            ))

        return columns

    else:
        for column in columns:
            if column['colName'] != data_columns[column['colIndex']]['colName']:
                raise ValueError("Dataset '{dataset_path}' has resource '{resource_id}' where column name '{metadata_name}' in metadata does not match column name '{data_name}' in data.".format(
                    dataset_path=dataset_path,
                    resource_id=resource_id,
                    metadata_name=column['colName'],
                    data_name=data_columns[column['colIndex']]['colName'],
                ))

            data_columns[column['colIndex']] = column

        return data_columns


def validate_target(problem_paths, dataset_path, problem_description, dataset_description, target, check_target_values):
    error = False

    try:
        for data_resource in dataset_description['dataResources']:
            if data_resource['resID'] == target['resID']:
                columns = get_all_columns(dataset_path, data_resource['resID'], data_resource)
                for column in columns:
                    if target['colName'] == column['colName'] or target['colIndex'] == column['colIndex']:
                        if not (target['colName'] == column['colName'] and target['colIndex'] == column['colIndex']):
                            print("ERROR: Problem {problem_paths} has a target '{target_index}' which does not match a column '{column_index}' in dataset '{dataset_path}' fully.".format(
                                problem_paths=problem_paths,
                                target_index=target['targetIndex'],
                                column_index=column['colIndex'],
                                dataset_path=dataset_path,
                            ))
                            error = True

                        if check_target_values:
                            error = validate_target_values(problem_paths, dataset_path, problem_description, data_resource, target) or error

                        break
                else:
                    raise KeyError("Cannot find column with column name '{column_name}' or column index '{column_index}'.".format(
                        column_name=target['colName'],
                        column_index=target['colIndex'],
                    ))

                break
        else:
            raise KeyError("Cannot find data resource with resource ID '{resource_id}'.".format(
                resource_id=target['resID'],
            ))

    except (IndexError, KeyError):
        print("ERROR: Problem {problem_paths} has target with index '{target_index}' which does not resolve.".format(
            problem_paths=problem_paths,
            target_index=target['targetIndex'],
        ))
        return True

    except ValueError as error:
        print("ERROR: {error}".format(
            error=error,
        ))
        return True

    return error


def canonical_dataset_description(dataset_description):
    dataset_description = copy.deepcopy(dataset_description)

    del dataset_description['about']['datasetID']
    if 'digest' in dataset_description['about']:
        del dataset_description['about']['digest']

    return dataset_description


def datasets_equal(first_dataset_path, second_dataset_path):
    if first_dataset_path == second_dataset_path:
        return True

    first_dataset_base_path = os.path.dirname(first_dataset_path)
    second_dataset_base_path = os.path.dirname(second_dataset_path)

    dir_comparison = deep_dircmp.DeepDirCmp(first_dataset_base_path, second_dataset_base_path, hide=[], ignore=[])

    different_files = dir_comparison.get_left_only_recursive() + dir_comparison.get_right_only_recursive() + dir_comparison.get_common_funny_recursive() + dir_comparison.get_diff_files_recursive()

    # This one can be different. And if it is different, we compare it elsewhere for allowed differences.
    if 'datasetDoc.json' in different_files:
        different_files.remove('datasetDoc.json')

    if different_files:
        print("ERROR: Dataset '{first_dataset_path}' and dataset '{second_dataset_path}' are not the same: {differences}".format(
            first_dataset_path=first_dataset_path,
            second_dataset_path=second_dataset_path,
            differences=different_files,
        ))
        return False

    return True


def validate_dataset_reference(dataset_id, dataset_descriptions, targets, problem_description_value, check_target_values):
    error = False

    if dataset_id not in dataset_descriptions:
        print("ERROR: Problem {problem_paths} is referencing unknown dataset '{dataset_id}'.".format(
            problem_paths=problem_description_value['paths'],
            dataset_id=dataset_id,
        ))
        error = True
    else:
        dataset_description_value = dataset_descriptions[dataset_id]
        dataset_description = dataset_description_value['description']
        for i, target in enumerate(targets):
            if target['targetIndex'] != i:
                print("ERROR: Problem {problem_paths} has target with invalid target index '{target_index}'.".format(
                    problem_paths=problem_description_value['paths'],
                    target_index=target['targetIndex'],
                ))
                error = True
            error = validate_target(problem_description_value['paths'], dataset_description_value['path'], problem_description_value['description'], dataset_description, target, check_target_values) or error

    return error


def map_dataset_id(dataset_id, dataset_view_map):
    for view_map in dataset_view_map:
        if view_map['from'] == dataset_id:
            return view_map['to']
    else:
        raise KeyError("Could not map '{dataset_id}' in dataset view map.".format(dataset_id=dataset_id))


def validate(dataset_descriptions, problem_descriptions):
    print("Validating all datasets and problems.")

    error = False

    dataset_description_groups = collections.defaultdict(list)

    for problem_description_value in problem_descriptions.values():
        problem_description = problem_description_value['description']
        for data in problem_description.get('inputs', {}).get('data', []):
            error = validate_dataset_reference(data['datasetID'], dataset_descriptions, data.get('targets', []), problem_description_value, True) or error

            if 'datasetViewMaps' in problem_description.get('inputs', {}).get('dataSplits', {}):
                if {'train', 'test', 'score'} != set(problem_description['inputs']['dataSplits']['datasetViewMaps'].keys()):
                    print("ERROR: Problem {problem_paths} has dataset view maps with invalid keys.".format(
                        problem_paths=problem_description_value['paths'],
                    ))
                    error = True
                else:
                    error = validate_dataset_reference(map_dataset_id(data['datasetID'], problem_description['inputs']['dataSplits']['datasetViewMaps']['train']), dataset_descriptions, data.get('targets', []), problem_description_value, True) or error

                    # Test and score splits do not have all values, so we do not validate target values there.
                    error = validate_dataset_reference(map_dataset_id(data['datasetID'], problem_description['inputs']['dataSplits']['datasetViewMaps']['test']), dataset_descriptions, data.get('targets', []), problem_description_value, False) or error
                    error = validate_dataset_reference(map_dataset_id(data['datasetID'], problem_description['inputs']['dataSplits']['datasetViewMaps']['score']), dataset_descriptions, data.get('targets', []), problem_description_value, False) or error

        if 'clustering' in problem_description['about']['taskKeywords']:
            for data in problem_description.get('inputs', {}).get('data', []):
                for target in data.get('targets', []):
                    if 'numClusters' not in target:
                        print("ERROR: Problem {problem_paths} is a clustering problem but is missing 'numClusters' in target '{target_index}'.".format(
                            problem_paths=problem_description_value['paths'],
                            target_index=target['targetIndex'],
                        ))
                        error = True

            if 'dataSplits' in problem_description['inputs'] and set(problem_description['inputs']['dataSplits'].keys()) - {'datasetViewMaps'}:
                print("ERROR: Problem {problem_paths} is a clustering problem with data splitting configuration, but it should not have one.".format(
                    problem_paths=problem_description_value['paths'],
                ))
                error = True

    for dataset_description_value in dataset_descriptions.values():
        dataset_description = dataset_description_value['description']

        dataset_id = dataset_description['about']['datasetID']

        for suffix in ['_TEST', '_TRAIN', '_SCORE']:
            if dataset_id.endswith(suffix):
                dataset_description_groups[dataset_id[:-len(suffix)]].append(dataset_description_value)
                break

    for problem_description_value in problem_descriptions.values():
        problem_description = problem_description_value['description']

        # If any clustering problem is using dataset splits, we validate those splits.
        if 'clustering' in problem_description['about']['taskKeywords']:
            for data in problem_description.get('inputs', {}).get('data', []):
                # We check this elsewhere.
                if data['datasetID'] not in dataset_descriptions:
                    continue

                dataset_id = data['datasetID']

                for suffix in ['_TEST', '_TRAIN', '_SCORE']:
                    if dataset_id.endswith(suffix):
                        base_dataset_id = dataset_id[:-len(suffix)]
                        break
                else:
                    base_dataset_id = dataset_id

                # There should always be at least one dataset.
                datasets = dataset_description_groups[base_dataset_id]
                if len(datasets) > 1:
                    first_dataset_path = datasets[0]['path']
                    for second_dataset_value in datasets[1:]:
                        second_dataset_path = second_dataset_value['path']
                        if not datasets_equal(first_dataset_path, second_dataset_path):
                            print("ERROR: Problem {problem_paths} is a clustering problem, but its data splits are not all the same, for example, {first_dataset_path} and {second_dataset_path}.".format(
                                problem_paths=problem_description_value['paths'],
                                first_dataset_path=first_dataset_path,
                                second_dataset_path=second_dataset_path,
                            ))
                            error = True
                            break

    for dataset_description_group in dataset_description_groups.values():
        first_dataset_description_value = dataset_description_group[0]
        first_dataset_description = canonical_dataset_description(first_dataset_description_value['description'])
        for dataset_description_value in dataset_description_group[1:]:
            dataset_description = canonical_dataset_description(dataset_description_value['description'])

            if first_dataset_description != dataset_description:
                print("ERROR: Dataset '{first_dataset_path}' and dataset '{dataset_path}' are not the same.".format(
                    first_dataset_path=first_dataset_description_value['path'],
                    dataset_path=dataset_description_value['path'],
                ))
                error = True

    return error


def search_directory(datasets_directory, known_dataset_descriptions, known_problem_descriptions, *, strict_naming=True):
    error = False

    datasets_directory = os.path.abspath(datasets_directory)

    for dirpath, dirnames, filenames in os.walk(datasets_directory, followlinks=True):
        if 'datasetDoc.json' in filenames:
            # Do not traverse further (to not parse "datasetDoc.json" if they
            # exists in raw data filename).
            dirnames[:] = []

            dataset_description_path = os.path.join(dirpath, 'datasetDoc.json')

            error = validate_dataset_description(dataset_description_path, known_dataset_descriptions, strict_naming=strict_naming) or error

        if 'problemDoc.json' in filenames:
            # We continue traversing further in this case.

            problem_description_path = os.path.join(dirpath, 'problemDoc.json')

            error = validate_problem_description(problem_description_path, known_problem_descriptions) or error

    return error


def configure_parser(parser: argparse.ArgumentParser, *, skip_arguments=()):
    if 'no_strict_naming' not in skip_arguments:
        parser.add_argument(
            '-n', '--no-strict-naming', default=True, action='store_false', dest='strict_naming',
            help="do not require strict naming convention",
        )
    if 'directories' not in skip_arguments:
        parser.add_argument(
            'directories', metavar='DIR', nargs='*', default=['.'],
            help="path to a directory with datasets, default is current directory",
        )


def handler(arguments):
    error = False

    known_dataset_descriptions = {}
    known_problem_descriptions = {}

    for datasets_directory in arguments.directories:
        error = search_directory(datasets_directory, known_dataset_descriptions, known_problem_descriptions, strict_naming=arguments.strict_naming) or error

    error = validate(known_dataset_descriptions, known_problem_descriptions) or error

    if error:
        print("There are ERRORS.")
        sys.exit(1)
    else:
        print("There are no errors.")


def main(argv):
    parser = argparse.ArgumentParser(description="Validate datasets.")
    configure_parser(parser)

    arguments = parser.parse_args(argv[1:])

    handler(arguments)


if __name__ == '__main__':
    main(sys.argv)
