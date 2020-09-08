import uuid
import numpy
import pandas as pd
from d3m.container import pandas as container_pandas
from d3m.container.dataset import Dataset
from d3m.metadata import base as metadata_base
from d3m.metadata.problem import Problem

from axolotl.utils.schemas import PROBLEM_DEFINITION


def make_unique_columns(data):
    """
    Parameters
    ----------
    data : pd.DataFrame
        A dataframe to fix the column names.

    Returns
    -------
    The original dataframe where the columns are strings and has a unique name/
    """
    seen_columns_name = {}
    column_names = []
    for column in data.columns:
        if column in seen_columns_name:
            column_name = str(column) + '_' + str(seen_columns_name[column])
            seen_columns_name[column] += 1
        else:
            seen_columns_name[column] = 0
            column_name = str(column)
        column_names.append(column_name)
    data.columns = column_names
    return data


def get_dataset(input_data, target_index=-2, index_column=-1, semantic_types=None, parse=False):
    """
    A function that has as input a dataframe, and generates a D3M dataset.

    Parameters
    ----------
    input_data : pd.DataFrame
        The dataframe to be converted to d3m Dataset.
    target_index : int
        The index of the target, if index is not present, it will be ignored.
    index_column : int
        The index of the index target, if not provided it will look for d3m index, if not generate one.
    semantic_types : Sequence[Sequence[str]]
        A list of semantic types to be applied. The sequence must be of the same length of
        the dataframe columns.
    parse :
        A flag to determine if the dataset will contain parsed columns. By default is set to fault
        to make it compatible with most of D3M current infrastructure.

    Returns
    -------
    A D3M dataset.
    """
    data = make_unique_columns(input_data.copy(deep=True))
    if semantic_types is None:
        semantic_types = [[] for i in range(len(data.columns))]
        for i, _type in enumerate(input_data.dtypes):
            if _type == float:
                semantic_types[i].append('http://schema.org/Float')
            elif _type == int:
                semantic_types[i].append('http://schema.org/Integer')

    resources = {}

    if 'd3mIndex' in data.columns:
        index_column = list(data.columns).index("d3mIndex")
    else:
        if index_column == -1:
            data.insert(0, 'd3mIndex', range(len(data)))
            semantic_types.insert(0, [])
            target_index += 1
            index_column = 0

    data = container_pandas.DataFrame(data)

    # remove this
    if not parse:
        data = data.astype(str)
    metadata = metadata_base.DataMetadata()

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
        if i == index_column:
            metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, i), {
                'name': column_name,
                'structural_type': numpy.int64,
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            })
        else:
            _structural_type = str
            if semantic_types[i]:
                _semantic_types = semantic_types[i]
                if 'http://schema.org/Float' in _semantic_types:
                    _structural_type = numpy.float64
                elif 'http://schema.org/Integer' in _semantic_types:
                    _structural_type = numpy.int64
            else:
                _semantic_types = ['https://metadata.datadrivendiscovery.org/types/UnknownType']

            if not parse:
                _structural_type = str
            if i == target_index:
                _semantic_types += ['https://metadata.datadrivendiscovery.org/types/SuggestedTarget']
            else:
                _semantic_types += ['https://metadata.datadrivendiscovery.org/types/Attribute']

            metadata = metadata.update(('learningData', metadata_base.ALL_ELEMENTS, i), {
                'name': column_name,
                'structural_type': _structural_type,
                'semantic_types': _semantic_types
            })

    dataset_id = str(uuid.uuid4())
    dataset_metadata = {
        'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        'structural_type': Dataset,
        'id': dataset_id,
        'name': dataset_id,
        'digest': str(uuid.uuid4()),
        'dimension': {
            'name': 'resources',
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
            'length': len(resources),
        },
    }

    metadata = metadata.update((), dataset_metadata)

    dataset = Dataset(resources, metadata)
    return dataset


def import_dataframe(data_frame, *, index_column=-1, semantic_types=None):
    """
    Function that transforms a dataframe into a dataset.

    data_frame : pd.DataFrame
        The input dataframe to be converted to d3m Dataset.
    index_column : int
        The index of the index column.
    semantic_types : Sequence[Sequence[str]]
        A list of semantic types to be applied. The sequence must be of the same length of
        the dataframe columns.

    Returns
    -------
    A D3M dataset.
    """
    data = get_dataset(input_data=data_frame, index_column=index_column, semantic_types=semantic_types)
    return data


def import_input_data(x, y=None, *, target_index=None, index_column=-1, semantic_types=None, parse=False):
    """
    Function that takes an np.array or a dataframe and convert them to a D3M dataset.

    x : Union[pd.DataFrame, np.array]
        Input features or the features with targets if target index is specified.
    y : Union[pd.DataFrame, np.array]
        input features or the features with targets if target index is specified.
   target_index : int
        The index of the target, if index is not present, it will be ignored.
    index_column : int
        The index of the index target, if not provided it will look for d3m index, if not generate one.
    semantic_types : Sequence[Sequence[str]]
        A list of semantic types to be applied. The sequence must be of the same length of
        the dataframe columns.
    parse :
        A flag to determine if the dataset will contain parsed columns. By default is set to fault
        to make it compatible with most of D3M current infrastructure.

    Returns
    -------
    A D3M dataset.
    """

    if y is not None and target_index is not None:
        print('Ignoring target index, using y as target')

    _target_index = -1
    if y is not None:
        _x = pd.DataFrame(x)
        _y = pd.DataFrame(y)
        input_data = pd.concat((_x, _y), axis=1)
        _target_index = len(_x.columns)
    elif target_index is not None:
        input_data = x
    else:
        raise ValueError('Targets (y) or target index should be provide')

    if _target_index != -1:
        target_index = _target_index
    data = get_dataset(input_data=input_data, target_index=target_index,
                       index_column=index_column, semantic_types=semantic_types, parse=parse)

    return data


def generate_problem_description(dataset, task=None, *, task_keywords=None, performance_metrics=None):
    """
    A function that simplifies the generation of a problem description.

    Parameters
    ----------
    dataset : Dataset
        Dataset to be use for pipeline search.
    task : str
        A string that represent the problem type, currently only supported: ``binary_classification`` and
        ``regression``.
    task_keywords : List[TaskKeyword]
        A list of TaskKeyword.
    performance_metrics: List[PerformanceMetric]
        A list of PerformanceMetric.

    Returns
    -------
    A Problem
    """
    dataset_id = dataset.metadata.query(())['id']
    problem_id = dataset_id + '_problem'
    schema = 'https://metadata.datadrivendiscovery.org/schemas/v0/problem.json'
    version = '4.0.0'

    target_column_index = None

    for i in range(dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS,))['dimension']['length']):
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in \
                dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i,))['semantic_types']:
            target_column_index = i
            break

    if target_column_index is None:
        raise ValueError('Input dataframe does not contains targets')

    inputs = {
        'dataset_id': dataset_id,
        'targets': [{
            'column_index': target_column_index,
            'column_name': dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i,))['name'],
            'resource_id': 'learningData',
            'target_index': 0
        }]
    }

    problem = None
    if task is None:
        if performance_metrics is not None and task_keywords is not None:
            problem = {
                'performance_metrics': performance_metrics,
                'task_keywords': task_keywords
            }
    else:
        if task in PROBLEM_DEFINITION:
            problem = PROBLEM_DEFINITION[task]
        else:
            raise ValueError(task + """ task is not supported in default definitions. 
            You can define your own task by specifying the task_keywords and performance metrics.""")

    problem_description = {
        'id': problem_id,
        'schema': schema,
        'version': version,
        'inputs': [inputs],
        'problem': problem
    }

    return Problem(problem_description)


def generate_dataset_problem(x, y=None, task=None, *, target_index=None, index_column=-1,
                             semantic_types=None, parse=False, task_keywords=None, performance_metrics=None):
    """
    Function that takes an np.array or a dataframe and convert them to a D3M dataset.

    x : Union[pd.DataFrame, np.array]
        Input features or the features with targets if target index is specified.
    y : Union[pd.DataFrame, np.array]
        input features or the features with targets if target index is specified.
    task : str
        A string that represent the problem type, currently only supported: ``binary_classification`` and
        ``regression``.
   target_index : int
        The index of the target, if index is not present, it will be ignored.
    index_column : int
        The index of the index target, if not provided it will look for d3m index, if not generate one.
    semantic_types : Sequence[Sequence[str]]
        A list of semantic types to be applied. The sequence must be of the same length of
        the dataframe columns.
    parse :
        A flag to determine if the dataset will contain parsed columns. By default is set to fault
        to make it compatible with most of D3M current infrastructure.
    task_keywords : List[TaskKeyword]
        A list of TaskKeyword.
    performance_metrics: List[PerformanceMetric]
        A list of PerformanceMetric.

    Returns
    -------
    dataset : Dataset
        A D3M dataset.
    problem_description : Problem
        A D3M problem.
    """
    dataset = import_input_data(x, y=y, target_index=target_index, index_column=index_column,
                                semantic_types=semantic_types, parse=parse)
    problem_description = generate_problem_description(dataset=dataset, task=task, task_keywords=task_keywords,
                                                       performance_metrics=performance_metrics)

    return dataset, problem_description
