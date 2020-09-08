import json
import os
import uuid

import copy
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import PrimitiveStep
from d3m.container import DataFrame
from d3m import utils as d3m_utils

from axolotl.predefined_pipelines import preprocessor
from axolotl.utils import pipeline as pipeline_utils, schemas as schemas_utils

__all__ = ['fetch', 'fetch_from_file']


def fetch(input_data, problem_description, predefined_path=None):
    if predefined_path is None:
        root = os.path.join(os.path.dirname(__file__), '../..')
        predefined_path = os.path.join(root, 'axolotl', 'utils', 'resources', 'default_pipelines.json')
    # ToDo should use yield
    pipelines = list()
    pipelines_from_file = fetch_from_file(problem_description, path=predefined_path)
    pipelines_from_preprocessors = _fetch_from_preprocessors(input_data, problem_description)
    for candiate in (
        pipelines_from_file,
        pipelines_from_preprocessors,
    ):
        pipelines.extend(candiate)
    return pipelines


def fetch_from_file(problem_description, path):
    # ToDo should use yield
    task_type, task_subtype, data_types, semi = _get_task_description(problem_description)

    pipelines = []
    with open(path) as file:
        possible_pipelines = json.load(file)
        with d3m_utils.silence():
            for task_type_in_file, pipeline_infos in possible_pipelines.items():
                if task_type_in_file == task_type:
                    for pipeline_info in pipeline_infos:
                        pipeline = pipeline_utils.load_pipeline(pipeline_info)
                        pipelines.append(pipeline)
    return pipelines


def _fetch_from_preprocessors(input_data, problem_description):
    task_type, task_subtype, data_types, semi = _get_task_description(problem_description)
    primitive_candidates = pipeline_utils.get_primitive_candidates(task_type, data_types, semi)

    mapped_task_type = schemas_utils.get_task_mapping(task_type)
    if mapped_task_type != task_type:
        primitive_candidates += pipeline_utils.get_primitive_candidates(mapped_task_type, data_types, semi)

    pipelines = []
    for primitive_info in primitive_candidates:
        if not check_primitive_dataframe_input(primitive_info):
            continue
        pps = preprocessor.get_preprocessor(
            input_data=input_data, problem=problem_description, treatment=primitive_info[1]
        )
        for pp in pps:
            pipeline_description = copy.deepcopy(pp.pipeline_description)
            pipeline_description.id = str(uuid.uuid4())
            pipeline = _complete_pipeline(
                pipeline_description=pipeline_description,
                dataframe_step=pp.dataset_to_dataframe_step,
                primitive_info=primitive_info,
                attributes=pp.attributes,
                targets=pp.targets,
                resolver=pp.resolver
            )
            pipelines.append(pipeline)
    return pipelines


def check_primitive_dataframe_input(primitive_info):
    primitive, _ = primitive_info
    primitive_arguments = primitive.metadata.query()['primitive_code']['arguments']
    if 'inputs' in primitive_arguments and primitive_arguments['inputs']['type'] == DataFrame:
        return True
    else:
        return False


def get_primitive(name):
    primitive = index.get_primitive(name)
    return primitive


def _complete_pipeline(pipeline_description, dataframe_step, attributes, targets, resolver, primitive_info):
    primitive, specific_primitive = primitive_info
    construct_prediction = 'd3m.primitives.data_transformation.construct_predictions.Common'
    construct_prediction_primitive = get_primitive(construct_prediction)

    _add_primitive_to_pipeline(pipeline_description, primitive, resolver, attributes, targets)
    _add_primitive_to_pipeline(pipeline_description, construct_prediction_primitive, resolver,
                               dataframe_step=dataframe_step)
    # Get the last step for the output
    last_step_idx = len(pipeline_description.steps) - 1
    output = pipeline_utils.int_to_step(last_step_idx)

    # Adding output step to the pieline
    pipeline_description.add_output(name='Predictions from the input dataset', data_reference=output)
    return pipeline_description


def _add_primitive_to_pipeline(pipeline_description, primitive, resolver, attributes=None, targets=None,
                               dataframe_step=None):
    step_model = PrimitiveStep(primitive=primitive, resolver=resolver)

    if dataframe_step is None:
        step_model.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
        step_model.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
    else:
        last_step_idx = len(pipeline_description.steps) - 1
        step_model.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                                data_reference=pipeline_utils.int_to_step(last_step_idx))
        step_model.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference=dataframe_step)
    step_model.add_output('produce')
    pipeline_description.add_step(step_model)


def _get_task_description(problem_description):
    task_description = schemas_utils.get_task_description(problem_description['problem']['task_keywords'])
    task_type = task_description['task_type']
    task_subtype = task_description['task_subtype']
    data_types = task_description['data_types']
    semi = task_description['semi']
    return task_type, task_subtype, data_types, semi
