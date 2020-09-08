import os
import pprint
import typing
import uuid
import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx
import pandas

import d3m
from d3m import container
from d3m import utils as d3m_utils
from d3m.container import utils as container_utils
from d3m.metadata import base as metadata_base
from d3m.metadata.pipeline import Pipeline, PlaceholderStep, PrimitiveStep, SubpipelineStep, get_pipeline, Resolver
from d3m.metadata.pipeline_run import PipelineRun
from d3m.metadata import problem as problem_module
from d3m.primitive_interfaces import base
from d3m.container.pandas import DataFrame


class PipelineResult:
    """
    A class that captures the output of multiple operations around the system.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline used for the run (fit/score)
    fitted_pipeline_id: str
        The id of the fitted pipeline used to produce the result.

    Attributes
    ----------
     pipeline: Pipeline
        Pipeline used for the run (fit/score)
     fitted_pipeline_id: str
        The id of the fitted pipeline used to produce the result.
     status: str
        A string representing the  status of the run (PENDING, RUNNING, COMPLETED, ERRORED)
     error: typing.Union[Exception, typing.List[Exception]]
        An error of list of errors occured during the execution of the pipeline or fitted pipeline.
     exposed_outputs: typing.Dict[str, typing.Any]
        A dictionary containing the name of te exposed output and the value, this could be a string
        of the path of the stored output or the object itself.
     output: container.DataFrame
        A dataframe of the pipeline output, this could be a string if the output is stored.
     pipeline_run
        A pipeline run, or the path where is stored.
     method_called: str
        The method that it was called while generating this result. (fit, produce)
     scores: pandas.DataFrame
        A dataframe containing the scores of the evaluated pipeline.
     rank: float
        The rank of the pipeline from 0 to 1, where 0 is the best.
    """
    def __init__(self, *, pipeline: Pipeline = None, fitted_pipeline_id: str = None):
        self.pipeline = pipeline
        self.fitted_pipeline_id: str = fitted_pipeline_id
        self.status: str = None
        self.error: typing.Union[Exception, typing.List[Exception]] = None
        self.exposed_outputs: typing.Dict[str, typing.Any] = None
        self.output: container.DataFrame = None
        self.pipeline_run = None
        self.method_called: str = None
        self.scores: pandas.DataFrame = None
        self.rank: float = None

    def __str__(self):
        string_representation = {}

        for name, value in self.__dict__.items():
            if not name.startswith('__') and not callable(name):
                if value is not None:
                    string_representation[name] = str(value)

        return pprint.pformat(string_representation).replace("\\n", "")

    def __repr__(self):
        base_string = 'PipelineResult'
        if self.pipeline is not None:
            base_string += ' pipeline_id:{}'.format(self.pipeline.id)

        if self.fitted_pipeline_id is not None:
            base_string += ' fitted_pipeline_id:{}'.format(self.fitted_pipeline_id)

        return base_string


class PrimitivesList:
    # root = os.path.dirname(__file__)
    # black_list = os.path.join(root, 'axolotl', 'utils', 'resources', 'blacklist.json')
    with open(os.path.join(os.path.dirname(__file__), 'resources', 'blocklist.json'), 'r') as file:
        BlockList = json.load(file)


class BlackListResolver(Resolver):
    """
    A resolver to resolve primitives and pipelines.

    It resolves primitives from available primitives on the system,
    and resolves pipelines from files in pipeline search paths.

    Attributes
    ----------
    strict_resolving : bool
        If resolved primitive does not fully match specified primitive reference, raise an exception?
    pipeline_search_paths : Sequence[str]
        A list of paths to directories with pipelines to resolve from.
        Their files should be named ``<pipeline id>.json`` or ``<pipeline id>.yml``.

    Parameters
    ----------
    strict_resolving : bool
        If resolved primitive does not fully match specified primitive reference, raise an exception?
    pipeline_search_paths : Sequence[str]
        A list of paths to directories with pipelines to resolve from.
        Their files should be named ``<pipeline id>.json`` or ``<pipeline id>.yml``.
    respect_environment_variable : bool
        Use also (colon separated) pipeline search paths from ``PIPELINES_PATH`` environment variable?
    """

    def __init__(self, black_list=PrimitivesList.BlockList, *, strict_resolving: bool = False, strict_digest: bool = False,
                 pipeline_search_paths: typing.Sequence[str] = None,
                 respect_environment_variable: bool = True, load_all_primitives: bool = True,
                 primitives_blocklist: typing.Collection[str] = None) -> None:
        super().__init__(strict_resolving=strict_resolving, strict_digest=strict_digest,
                         pipeline_search_paths=pipeline_search_paths,
                         respect_environment_variable=respect_environment_variable,
                         load_all_primitives=load_all_primitives, primitives_blocklist=primitives_blocklist)
        self.black_list = black_list
        if len(black_list) == 0:
            self.black_list = None

    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        if not self._primitives_loaded:
            self._primitives_loaded = True

            d3m.index.load_all(blacklist=self.black_list)

        return d3m.index.get_primitive_by_id(primitive_description['id'])


def load_pipeline(pipeline_file: typing.Union[str, typing.Dict]):
    """
    Load pipeline from a pipeline URI

    Parameters
    ----------
    pipeline_file: Union[str, dict]
        The URI pointing to a json file of pipeline or dict of string that is a pipeline

    Returns
    -------
    pipeline: Pipeline
        An object of Pipeline

    """
    if isinstance(pipeline_file, dict):
        try:
            with d3m_utils.silence():
                pipeline = Pipeline.from_json_structure(pipeline_file)
        except:
            pipeline = None
    else:
        with d3m_utils.silence():
            pipeline = get_pipeline(pipeline_path=pipeline_file, load_all_primitives=False)
    return pipeline


def save_pipeline(pipeline, path, *, rank=None):
    """
    A function that make a copy of an already scored pipeline to scored directory according with specifications.

    Parameters
    ----------
    pipeline : Pipeline
        A pipeline to be save into the path
    path: str
        Path where the pipeline will be stored
    rank : float
        A float that represents the rank of the pipeline.
    """

    pipeline_path = os.path.join(path, '{}.json'.format(pipeline.id))

    with open(pipeline_path, 'w') as file:
        pipeline.to_json(file, indent=2, sort_keys=True, ensure_ascii=False)

    if rank is not None:
        rank_path = os.path.join(path, '{}.rank'.format(pipeline.id))
        with open(rank_path, 'w') as file:
            file.write('{rank}'.format(rank=rank))


def save_pipeline_run(pipeline_run, path):
    """
    A function that make a copy of an already scored pipeline to scored directory according with specifications.

    Parameters
    ----------
    pipeline_run : PipelineRun
        A pipeline_run to be save into the path
    path: str
        Path where the pipeline_run will be stored

    Returns
    -------
    pipeline_run_path : str
        Path where the pipeline_run is stored.
    """

    if pipeline_run is None:
        return

    if isinstance(pipeline_run, list):
        first = True
        pipeline_run_path = os.path.join(path, '{}.yml'.format(pipeline_run[0].pipeline['id']))
        with d3m_utils.silence():
            with open(pipeline_run_path, 'w') as file:
                for run in pipeline_run:
                    run.to_yaml(file, appending=not first)
                    first = False
    else:
        pipeline_run_path = os.path.join(path, '{}.yml'.format(pipeline_run.pipeline['id']))
        with d3m_utils.silence():
            with open(pipeline_run_path, 'w') as file:
                pipeline_run.to_yaml(file)

    return pipeline_run_path


def save_exposed_values(values, output_id, output_dir):
    """
    A function to save the exposed values of a PipelineResult.

    Parameters
    ----------
    values : Union[dict[str, container], container]
        A container to be stored into the path
    output_id : str
        An id that identify the values.
    output_dir : str
        The path where the values are going to be store.

    Returns
    -------
    A dict of names and stored paths.

    """
    output_paths = {}
    output_path = os.path.join(output_dir, output_id)
    unique_id = str(uuid.uuid4())

    def get_file_path(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        file_path = ""
        if 'data.csv' in files:
            file_path = os.path.join(path, 'data.csv')
        elif 'datasetDoc.json' in files:
            file_path = os.path.join(path, 'datasetDoc.json')
        return file_path

    if isinstance(values, dict):
        for name, value in values.items():
            _output_path = os.path.join(output_path, output_id, unique_id, name)
            container_utils.save_container(value, _output_path)
            output_paths[name] = get_file_path(_output_path)
    else:
        _output_path = os.path.join(output_path, output_id, unique_id, 'output')
        container_utils.save_container(values, _output_path)
        output_paths['output'] = get_file_path(_output_path)

    return output_paths


def plot_pipeline(pipeline):
    figure(num=None, figsize=(10, 12), dpi=80, facecolor='w', edgecolor='k')
    graph, nodes_info = get_pipeline_graph(pipeline)

    the_table = plt.table(cellText=nodes_info, colWidths=[0.05, 0.5], colLabels=['Step', 'Primitive'], loc='right')
    the_table.set_fontsize(25)
    the_table.scale(2, 1)
    pos = nx.kamada_kawai_layout(graph, scale=3)
    grafo_labels = nx.get_edge_attributes(graph, 'label')
    edges_label = nx.draw_networkx_edge_labels(graph, pos, edge_labels=grafo_labels, font_size=7)
    nx.draw(graph, pos=pos, node_size=900, alpha=0.5, font_size=16, edges_label=edges_label, with_labels=True, scale=5)


def __get_header(index, step):
    if isinstance(step, PrimitiveStep):
        header = 'steps.' + str(index) + '  -  ' + step.primitive.metadata.query()['python_path']
    elif isinstance(step, PlaceholderStep):
        header = 'steps.' + str(index) + '  -  ' + 'PlaceHolderStep'
    elif isinstance(step, SubpipelineStep):
        header = 'steps.' + str(index) + '  -  ' + 'SubPipeline'
    return header


def get_pipeline_graph(pipeline):
    graph = nx.DiGraph()
    nodes_info = []

    for i in range(0, len(pipeline.steps)):
        nodes_info.append([str(i), pipeline.steps[i].primitive.metadata.query()['python_path']])

        if isinstance(pipeline.steps[i], PrimitiveStep) or isinstance(pipeline.steps[i], PlaceholderStep):
            target = i
            graph.add_node(target)
            for argument in pipeline.steps[i].arguments.keys():
                data = pipeline.steps[i].arguments[argument]['data']
                if 'input' in data:
                    source = 'inputs'
                else:
                    index = int(data.split('.')[1])
                    source = index
                label = argument + '-' + data
                graph.add_edge(source, target, label=label)

            for hp in pipeline.steps[i].hyperparams.keys():
                if pipeline.steps[i].hyperparams[hp]['type'] == metadata_base.ArgumentType.PRIMITIVE:
                    index = pipeline.steps[i].hyperparams[hp]['data']
                    source = index
                    label = 'Step {} hyperparam - {}'.format(i, hp)
                    graph.add_edge(source, target, label=label)
        else:
            # TODO add support here for subpipelines
            continue

    for i in range(0, len(pipeline.outputs)):
        index = int(pipeline.outputs[i]['data'].split('.')[1])
        source = index
        label = 'outputs.{}'.format(i)
        graph.add_edge(source, 'output', label=label)

    return graph, nodes_info


def infer_primitive_family(task_type: str, data_types: typing.Iterable, is_semi: bool = False) -> typing.Optional[str]:
    """
    Infer target primitive family by task and data_types

    Parameters
    ----------
    task_type: str
        The task type
    data_types: typing.Iterable
        The data types
    is_semi: bool
        Is semi supervised probelm

    Returns
    -------
    str
        The primitive family
    """

    #TODO temp solution
    if problem_module.TaskKeyword.CLASSIFICATION == task_type and \
            problem_module.TaskKeyword.TIME_SERIES in data_types and \
            problem_module.TaskKeyword.GROUPED in data_types:
        return metadata_base.PrimitiveFamily.CLASSIFICATION
    if problem_module.TaskKeyword.CLASSIFICATION == task_type and \
            problem_module.TaskKeyword.TIME_SERIES in data_types:
        return metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name
    if problem_module.TaskKeyword.FORECASTING and problem_module.TaskKeyword.TIME_SERIES in data_types:
        return metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING.name
    if problem_module.TaskKeyword.CLASSIFICATION == task_type and is_semi:
        return metadata_base.PrimitiveFamily.SEMISUPERVISED_CLASSIFICATION.name
    if problem_module.TaskKeyword.IMAGE in data_types:
        return metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING.name
    if problem_module.TaskKeyword.VIDEO in data_types:
        return metadata_base.PrimitiveFamily.DIGITAL_SIGNAL_PROCESSING.name

    return task_type


def check_black_list(primitive_name: str, extra_block: typing.List=[]) -> bool:
    """
    Check if the primitive is in the black list, which is from `LIST.BlACK_LIST`

    Parameters
    ----------
    primitive_name: str
        The name of the primitive

    Returns
    -------
    bool

    """
    banned_terms = PrimitivesList.BlockList + extra_block
    for banned_element in banned_terms:
        if banned_element in primitive_name:
            return True
    return False


def get_primitive_candidates(task_type: str, data_types: typing.Iterable, semi: bool,
                             extra_block: typing.List=[]) -> typing.List:
    """
    Get a list of primitive candidates related to the task type except those primitives in `BLACK_LIST`

    Parameters
    ----------
    task_type: str
        The task type
    data_types: typing.Iterable
        The data types
    semi: bool
        Is it semi-supervised problem

    Returns
    -------
    list
        A list of primitives
    """
    specific_task = infer_primitive_family(task_type, data_types, semi)
    primitives_path = d3m.index.search()
    primitives = list()
    for primitive_path in primitives_path:
        if check_black_list(primitive_path, extra_block):
            continue
        try:
            with d3m_utils.silence():
                primitive = d3m.index.get_primitive(primitive_path)
            primitive_family = primitive.metadata.query()['primitive_family'].name
            if primitive_family == task_type:
                primitives.append((primitive, task_type))
            elif primitive_family == specific_task:
                primitives.append((primitive, specific_task))
        # TODO what exception?
        except Exception as e:
            continue
    return primitives


def int_to_step(n_step: int) -> str:
    """
    Convert the step number to standard str step format

    Parameters
    ----------
    n_step: int

    Returns
    -------
    str
        str format in "steps.<n_step>.produce"
    """
    return 'steps.' + str(n_step) + '.produce'


def get_primitives(primitives_dict):
    """
    A function that loads and returns a dictionary of primitives

    Parameters
    ----------
    primitives_dict: dict[str, str]
        A dictionary that contains the alias and the primitives to load.

    Returns
    -------
    loaded_primitives_dict: dict[str, str]
        A dictionary containing the aliases and the loaded primitives.
    """
    loaded_primitives_dict = {}
    for primitive_name in primitives_dict.keys():
        loaded_primitives_dict[primitive_name] = d3m.index.get_primitive(primitives_dict[primitive_name])
    return loaded_primitives_dict


def get_tabular_resource_id(dataset):
    """
    A function that retrieves the main resource id

    Parameters
    ----------
    dataset: Dataset
        A dataset.

    Returns
    -------
    resource_id: str
        An id of the main resource.
    """

    resource_id = None
    for dataset_resource_id in dataset.keys():
        if dataset.metadata.has_semantic_type((dataset_resource_id,),
                                              'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'):
            resource_id = dataset_resource_id
            break

    if resource_id is None:
        tabular_resource_ids = [dataset_resource_id for dataset_resource_id, dataset_resource in dataset.items() if
                                isinstance(dataset_resource, container.DataFrame)]
        if len(tabular_resource_ids) == 1:
            resource_id = tabular_resource_ids[0]

    if resource_id is None:
        resource_id = 'learningData'

    return resource_id


def query_multiple_terms(metadata, list_queries):
    data = metadata.query()
    valid_queries = []
    for query in list_queries:
        if query in data:
            valid_queries.append(query)
            data = data[query]
        else:
            break
    if len(valid_queries) == len(list_queries):
        return data


def filter_primitives_by_dataframe_input(primitive_info):
    primitives_dataframe_input = []
    for info in primitive_info:
        primitive, task = info
        arguments = query_multiple_terms(
            primitive.metadata, ['primitive_code', 'class_type_arguments'])

        has_dataframe_arguments = True
        for argument, value in arguments.items():
            if argument == 'Params' or argument == 'Hyperparams':
                continue
            else:
                if value != DataFrame:
                    has_dataframe_arguments = False
                    break
        if has_dataframe_arguments:
            primitives_dataframe_input.append(info)

    return primitives_dataframe_input

