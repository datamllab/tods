#!/usr/bin/env python3

import argparse
import enum
import json
import os
import os.path
import sys

import numpy
import pandas

from d3m import container


class DatasetType(enum.Enum):
    COUNTS_PER_USER = 1
    COMMENTS_PER_POST = 2
    HAS_USER_MADE_COMMENT_ON_POST = 3


def pareto_choice(random_state, array, size):
    # 80/20 rule.
    a = 1.161

    p = random_state.pareto(a, size=len(array)) + 1
    p /= numpy.sum(p)

    return random_state.choice(array, size=size, replace=True, p=p)


def generate_main_resources(random_state, resources, size):
    users_count = size
    posts_count = size * 10
    comments_count = size * 10

    user_ids = numpy.array(range(users_count))
    post_ids = numpy.array(range(posts_count))
    comment_ids = numpy.array(range(comments_count))

    users = container.DataFrame({
        'id': user_ids,
        'name': [f'User {i}' for i in range(users_count)],
    })

    posts = container.DataFrame({
        'id': post_ids,
        'author_id': pareto_choice(random_state, user_ids, posts_count),
        'post': [f'Post {i}' for i in range(posts_count)],
    })

    comments = container.DataFrame({
        'id': comment_ids,
        'post_id': pareto_choice(random_state, post_ids, comments_count),
        'author_id': pareto_choice(random_state, user_ids, comments_count),
        'comment': [f'Comment {i}' for i in range(comments_count)],
    })

    resources.update({'users': users, 'posts': posts, 'comments': comments})


def generate_learning_data_counts_per_user(random_state, resources):
    user_ids = resources['users'].loc[:, 'id']
    users_count = len(user_ids)
    posts = resources['posts']
    comments = resources['comments']

    learning_data = container.DataFrame({
        'd3mIndex': numpy.array(range(users_count)),
        'user_id': user_ids,
        'posts_count': [(posts.loc[:, 'author_id'] == user_id).sum() for user_id in user_ids],
        'comments_count': [(comments.loc[:, 'author_id'] == user_id).sum() for user_id in user_ids],
    })

    resources['learningData'] = learning_data


def generate_learning_data_comments_per_post(random_state, resources):
    post_ids = resources['posts'].loc[:, 'id']
    posts_count = len(post_ids)
    comments = resources['comments']

    learning_data = container.DataFrame({
        'd3mIndex': numpy.array(range(posts_count)),
        'post_id': post_ids,
        'comments_count': [(comments.loc[:, 'post_id'] == post_id).sum() for post_id in post_ids],
    })

    resources['learningData'] = learning_data


def generate_learning_data_has_user_made_comment_on_post(random_state, resources):
    user_ids = resources['users'].loc[:, 'id']
    post_ids = resources['posts'].loc[:, 'id']
    users_count = len(user_ids)
    comments = resources['comments']

    authors_and_posts = comments.loc[:, ['author_id', 'post_id']]

    authors_and_posts_set = set(authors_and_posts.itertuples(index=False, name=None))

    data = {
        'user_id': [],
        'post_id': [],
        'made_comment': [],
    }

    for author_id, post_id in authors_and_posts.sample(n=users_count, random_state=random_state).itertuples(index=False, name=None):
        data['user_id'].append(author_id)
        data['post_id'].append(post_id)
        data['made_comment'].append('yes')

    for user_id in random_state.permutation(user_ids):
        for post_id in random_state.permutation(post_ids):
            if (user_id, post_id) in authors_and_posts_set:
                continue

            data['user_id'].append(user_id)
            data['post_id'].append(post_id)
            data['made_comment'].append('no')

            if len(data['user_id']) == 2 * users_count:
                break

        if len(data['user_id']) == 2 * users_count:
            break

    assert len(data['user_id']) == 2 * users_count

    data = container.DataFrame(data)
    data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    index = container.DataFrame({
        'd3mIndex': numpy.array(range(len(data))),
    })

    resources['learningData'] = container.DataFrame(pandas.concat([index, data], axis=1))


def update_metadata_main_resources(dataset, dataset_id, dataset_type, size, random_seed):
    dataset.metadata = dataset.metadata.update((), {
        'id': dataset_id,
        'name': f"Database dataset of type {dataset_type}",
        'description': f"Database dataset of type {dataset_type}, size {size}, random seed {random_seed}",
    })

    dataset.metadata = dataset.metadata.update_column(0, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Integer'],
    }, at=('users',))
    dataset.metadata = dataset.metadata.update_column(1, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Text'],
    }, at=('users',))

    dataset.metadata = dataset.metadata.update_column(0, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Integer'],
    }, at=('posts',))
    dataset.metadata = dataset.metadata.update_column(1, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'users',
            'column_index': 0,
        },
    }, at=('posts',))
    dataset.metadata = dataset.metadata.update_column(2, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Text'],
    }, at=('posts',))

    dataset.metadata = dataset.metadata.update_column(0, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Integer'],
    }, at=('comments',))
    dataset.metadata = dataset.metadata.update_column(1, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'posts',
            'column_index': 0,
        },
    }, at=('comments',))
    dataset.metadata = dataset.metadata.update_column(2, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'users',
            'column_index': 0,
        },
    }, at=('comments',))
    dataset.metadata = dataset.metadata.update_column(3, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Text'],
    }, at=('comments',))


def update_metadata_counts_per_user(dataset):
    dataset.metadata = dataset.metadata.update_column(0, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Integer'],
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(1, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'users',
            'column_index': 0,
        },
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(2, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'http://schema.org/Integer'],
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(3, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'http://schema.org/Integer'],
    }, at=('learningData',))


def update_metadata_comments_per_post(dataset):
    dataset.metadata = dataset.metadata.update_column(0, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Integer'],
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(1, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'posts',
            'column_index': 0,
        },
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(2, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'http://schema.org/Integer'],
    }, at=('learningData',))


def update_metadata_has_user_made_comment_on_post(dataset):
    dataset.metadata = dataset.metadata.update_column(0, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Integer'],
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(1, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'users',
            'column_index': 0,
        },
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(2, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute', 'http://schema.org/Integer'],
        'foreign_key': {
            'type': 'COLUMN',
            'resource_id': 'posts',
            'column_index': 0,
        },
    }, at=('learningData',))
    dataset.metadata = dataset.metadata.update_column(3, {
        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'http://schema.org/Boolean'],
    }, at=('learningData',))


def handler(arguments):
    random_state = numpy.random.RandomState(arguments.random_seed)

    resources = {}
    generate_main_resources(random_state, resources, arguments.size)

    if arguments.dataset_type == DatasetType.COUNTS_PER_USER:
        generate_learning_data_counts_per_user(random_state, resources)

    elif arguments.dataset_type == DatasetType.COMMENTS_PER_POST:
        generate_learning_data_comments_per_post(random_state, resources)

    elif arguments.dataset_type == DatasetType.HAS_USER_MADE_COMMENT_ON_POST:
        generate_learning_data_has_user_made_comment_on_post(random_state, resources)

    else:
        raise ValueError(f"Unknown dataset type: {arguments.dataset_type.name}")

    dataset = container.Dataset(resources, generate_metadata=True)
    update_metadata_main_resources(dataset, arguments.dataset_id, arguments.dataset_type.name, arguments.size, arguments.random_seed)

    if arguments.dataset_type == DatasetType.COUNTS_PER_USER:
        update_metadata_counts_per_user(dataset)

    elif arguments.dataset_type == DatasetType.COMMENTS_PER_POST:
        update_metadata_comments_per_post(dataset)

    elif arguments.dataset_type == DatasetType.HAS_USER_MADE_COMMENT_ON_POST:
        update_metadata_has_user_made_comment_on_post(dataset)

    else:
        raise ValueError(f"Unknown dataset type: {arguments.dataset_type.name}")

    dataset_output_uri = 'file://' + os.path.join(os.path.abspath(arguments.output_dir), arguments.dataset_id, 'datasetDoc.json')

    dataset.save(dataset_output_uri)

    os.makedirs(os.path.join(os.path.abspath(arguments.output_dir), arguments.problem_id))

    with open(os.path.join(os.path.abspath(arguments.output_dir), arguments.problem_id, 'problemDoc.json'), 'x', encoding='utf8') as problem_file:
        if arguments.dataset_type == DatasetType.COUNTS_PER_USER:
            task_keywords = ['regression', 'multivariate']
            metric = {
                'metric': 'rootMeanSquaredError',
            }
            targets = [
                {
                    'targetIndex': 0,
                    'resID': 'learningData',
                    'colIndex': 2,
                    'colName': 'posts_count',
                },
                {
                    'targetIndex': 1,
                    'resID': 'learningData',
                    'colIndex': 3,
                    'colName': 'comments_count',
                },
            ]
        elif arguments.dataset_type == DatasetType.COMMENTS_PER_POST:
            task_keywords = ['regression', 'univariate']
            metric = {
                'metric': 'rootMeanSquaredError',
            }
            targets = [
                {
                    'targetIndex': 0,
                    'resID': 'learningData',
                    'colIndex': 2,
                    'colName': 'comments_count',
                },
            ]
        elif arguments.dataset_type == DatasetType.HAS_USER_MADE_COMMENT_ON_POST:
            task_keywords = ['classification', 'binary']
            metric = {
                'metric': 'f1',
                'posLabel': 'yes',
            }
            targets = [
                {
                    'targetIndex': 0,
                    'resID': 'learningData',
                    'colIndex': 3,
                    'colName': 'made_comment',
                },
            ]

        json.dump({
            'about': {
                'problemID': arguments.problem_id,
                'problemName': f"Database problem of type {arguments.dataset_type.name}",
                'taskKeywords': task_keywords,
                'problemSchemaVersion': '4.0.0',
            },
            'inputs': {
                'data': [
                    {
                        'datasetID': arguments.dataset_id,
                        'targets': targets,
                    },
                ],
                'performanceMetrics': [
                    metric,
                ],
            },
            'expectedOutputs': {
                'predictionsFile': 'predictions.csv',
                'scoresFile': 'scores.csv',
            },
        }, problem_file, indent=2)


def main(argv):
    parser = argparse.ArgumentParser(description="Generate database datasets.")

    parser.add_argument(
        '--dataset-type', choices=[dataset_type.name for dataset_type in DatasetType], action='store', required=True,
        help="what type of dataset to generate",
    )
    parser.add_argument(
        '--dataset-id', action='store', required=True,
        help="dataset ID to use",
    )
    parser.add_argument(
        '--problem-id', action='store', required=True,
        help="problem ID to use",
    )
    parser.add_argument(
        '--random-seed', type=int, action='store', default=0,
        help="random seed to use",
    )
    parser.add_argument(
        '--size', type=int, action='store', default=1000,
        help="size of dataset to generate",
    )
    parser.add_argument(
        '--output-dir', action='store', default='.',
        help="directory where to store generated dataset and problem, default is current directory",
    )

    arguments = parser.parse_args(argv[1:])

    arguments.dataset_type = DatasetType[arguments.dataset_type]

    handler(arguments)


if __name__ == '__main__':
    main(sys.argv)
