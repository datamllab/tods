import pandas as pd
from pandas import DataFrame,Series
from ctypes import alignment
import uuid
from d3m.metadata import base as metadata_base

from typing import List, Optional
from collections import OrderedDict
import uuid

import numpy as np

import os

from d3m.container import DataFrame as d3m_dataframe
from d3m.primitive_interfaces.base import Hyperparams
from d3m.metadata import base as metadata_base

import argparse

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from sklearn import datasets

primitive_families = {'data_transform': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
                      'data_preprocessing': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
                      'data_validate': metadata_base.PrimitiveFamily.DATA_VALIDATION,
                      'data_cleaning': metadata_base.PrimitiveFamily.DATA_CLEANING,
                      'anomaly_detect': metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
                      'feature_construct': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
                      'evaluation': metadata_base.PrimitiveFamily.EVALUATION,
                      'feature_extract': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
                      }
algorithm_type = {'file_manipulate': metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
                  'data_denormalize': metadata_base.PrimitiveAlgorithmType.DATA_DENORMALIZATION,
                  'data_split': metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,
                  'k_fold': metadata_base.PrimitiveAlgorithmType.K_FOLD,
                  'cross_validate': metadata_base.PrimitiveAlgorithmType.CROSS_VALIDATION,
                  'identity': metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
                  'data_convert': metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
                  'holdout': metadata_base.PrimitiveAlgorithmType.HOLDOUT,
}


default_primitive = {'data_processing': [],
                     'timeseries_processing': [],
                     'feature_analysis': [['statistical_maximum',None]],
                     'detection_algorithm': [['pyod_ae',
                                              {"contamination": 0.1,
                                               "use_semantic_types":True,
                                               "use_columns": [2]}]],
}

def construct_primitive_metadata(module, name, id, primitive_family, hyperparams=None, algorithm=None, flag_hyper=False,
                                 description=None):
    if algorithm == None:
        temp = [metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE]
    else:
        temp = []
        for alg in algorithm:
            temp.append(algorithm_type[alg])
    meta_dict = {
        "__author__ ": "DATA Lab @ Texas A&M University",
        'version': '0.3.0',
        'name': description,
        'python_path': 'd3m.primitives.tods.' + module + '.' + name,
        'source': {
            'name': "DATA Lab @ Texas A&M University",
            'contact': 'mailto:khlai037@tamu.edu',
        },
        'algorithm_types': temp,
        'primitive_family': primitive_families[primitive_family],  # metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, id)),

    }
    # if name1 != None:
    # meta_dict['name'] = name1
    if hyperparams != None:
        if flag_hyper == False:
            meta_dict['hyperparams_to_tune'] = hyperparams
        else:
            meta_dict['hyperparameters_to_tune'] = hyperparams
    metadata = metadata_base.PrimitiveMetadata(meta_dict, )
    return metadata

def datapath_to_dataset(path, target_index):
    df = pd.read_csv(path)
    return generate_dataset(df, target_index)

def load_pipeline(pipeline_path):  # pragma: no cover
    """Load a pipeline given a path
    Args:
        pipeline_path (str): The path to a pipeline file
    Returns:
        pipeline
    """
    from axolotl.utils import pipeline as pipeline_utils
    pipeline = pipeline_utils.load_pipeline(pipeline_path)
    return pipeline


def generate_dataset(df, target_index, system_dir=None):  # pragma: no cover
    """Generate dataset
    Args:
        df (pandas.DataFrame): dataset
        target_index (int): The column index of the target
        system_dir (str): Where the systems will be stored
    returns:
        dataset
    """
    from axolotl.utils import data_problem
    dataset = data_problem.import_input_data(df, target_index=target_index, media_dir=system_dir)
    return dataset

def generate_problem(dataset, metric):  # pragma: no cover
    """Generate dataset
    Args:
        dataset: dataset
        metric (str): `F1` for computing F1 on label 1, 'F1_MACRO` for
            macro-F1 on both 0 and 1
    returns:
        problem_description
    """
    from axolotl.utils import data_problem
    from d3m.metadata.problem import TaskKeyword, PerformanceMetric
    if metric == 'F1':
        performance_metrics = [{'metric': PerformanceMetric.F1, 'params': {'pos_label': '1'}}]
    elif metric == 'F1_MACRO':
        performance_metrics = [{'metric': PerformanceMetric.F1_MACRO, 'params': {}}]
    elif metric == 'RECALL':
        performance_metrics = [{'metric': PerformanceMetric.RECALL, 'params': {'pos_label': '1'}}]
    elif metric == 'PRECISION':
        performance_metrics = [{'metric': PerformanceMetric.PRECISION, 'params': {'pos_label': '1'}}]
    elif metric == 'PRECISION':
        performance_metrics = [{'metric': PerformanceMetric.PRECISION, 'params': {'pos_label': '1'}}]
    elif metric == 'ALL':
        performance_metrics = [{'metric': PerformanceMetric.PRECISION, 'params': {'pos_label': '1'}},
                               {'metric': PerformanceMetric.RECALL, 'params': {'pos_label': '1'}},
                               {'metric': PerformanceMetric.F1_MACRO, 'params': {}},
                               {'metric': PerformanceMetric.F1, 'params': {'pos_label': '1'}}]
    else:
        raise ValueError('The metric {} not supported.'.format(metric))
    problem_description = data_problem.generate_problem_description(dataset=dataset,
                                                                    task_keywords=[TaskKeyword.ANOMALY_DETECTION, ],
                                                                    performance_metrics=performance_metrics)

    return problem_description


def evaluate_pipeline(dataset, pipeline, metric='F1', seed=0):  # pragma: no cover
    """Evaluate a Pipeline
    Args:
        dataset: A dataset
        pipeline: A pipeline
        metric (str): `F1` for computing F1 on label 1, 'F1_MACRO` for
            macro-F1 on both 0 and 1
        seed (int): A random seed
    Returns:
        pipeline_result
    """
    from axolotl.utils import schemas as schemas_utils
    from axolotl.backend.simple import SimpleRunner
    problem_description = generate_problem(dataset, metric)
    data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
    scoring_pipeline = schemas_utils.get_scoring_pipeline()
    data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']
    metrics = problem_description['problem']['performance_metrics']

    backend = SimpleRunner(random_seed=seed) 
    pipeline_result = backend.evaluate_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset],
                                                metrics=metrics,
                                                data_preparation_pipeline=data_preparation_pipeline,
                                                scoring_pipeline=scoring_pipeline,
                                                data_preparation_params=data_preparation_params)
    return pipeline_result


# Build pipeline

Inputs = d3m_dataframe
Outputs = d3m_dataframe


def build_pipeline(config):
    """
    Build a pipline based on the user defined config
    Args:
        config:
            A user defined config specifying various primitives/hyperparams
            to be used for pipeline construction
    """
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')


    #Step 0: dataset_to_dataframe
    step_0 = build_step(arguments={'inputs': 'inputs.0'},
                        primitive_path='data_processing.dataset_to_dataframe')
    pipeline_description.add_step(step_0)


    #Step 1: column_parser
    step_1 = build_step(arguments={'inputs':'steps.0.produce'},
                        primitive_path='data_processing.column_parser')
    pipeline_description.add_step(step_1)

    #Step 2: extract_columns_by_semantic_types(attributes)
    step_2 = build_step(arguments={'inputs': 'steps.1.produce'},
                        primitive_path='data_processing.extract_columns_by_semantic_types',
                        hyperparams={'semantic_types':['https://metadata.datadrivendiscovery.org/types/Attribute']})
    pipeline_description.add_step(step_2)


    #Step 3: extract_columns_by_semantic_types(targets)
    # step_3 = build_step(arguments={'inputs':'steps.0.produce'},
    #                     primitive_path='data_processing.extract_columns_by_semantic_types',
    #                     hyperparams={'semantic_types':['https://metadata.datadrivendiscovery.org/types/TrueTarget']})
    # pipeline_description.add_step(step_3)

    attributes = 'steps.2.produce'
    targets = 'steps.3.produce'

    counter = 2
    flag = 1


    #Step 4: time series processing
    timeseries_processing_methods = get_primitive_list(config,'timeseries_processing')
    timeseries_processing_hyperparams = get_primitives_hyperparam_list(config,'timeseries_processing')
    for i in range(len(timeseries_processing_methods)):
        
        step_processing = build_step(arguments={'inputs':'steps.' + str(counter) + '.produce'},
                                     primitive_path='timeseries_processing.'+ timeseries_processing_methods[i],
                                     hyperparams=timeseries_processing_hyperparams[i])
        pipeline_description.add_step(step_processing)
        counter += 1
 
        
    # Step 5: Feature analysis
    feature_analysis_methods = get_primitive_list(config,'feature_analysis')
    feature_analysis_hyperparams = get_primitives_hyperparam_list(config,'feature_analysis')

    for i in range(len(feature_analysis_methods)):
        step_processing = build_step(arguments={'inputs': 'steps.' + str(counter) + '.produce'},
                                     primitive_path='feature_analysis.'+feature_analysis_methods[i],
                                     hyperparams=feature_analysis_hyperparams[i])
        pipeline_description.add_step(step_processing)
        counter += 1



    #Step 6: detectionn algorithm
    detection_algorithm_methods = get_primitive_list(config,'detection_algorithm')
    detection_algorithm_hyperparams = get_primitives_hyperparam_list(config,'detection_algorithm')

    step_5 = build_step(arguments={'inputs': 'steps.' + str(counter) + '.produce'},
                        primitive_path='detection_algorithm.'+detection_algorithm_methods[0],
                        hyperparams=detection_algorithm_hyperparams[0])
    # step_5 = build_step(arguments={'inputs': 'steps.' + str(counter) + '.produce'},
    #                 primitive_path='detection_algorithm.'+detection_algorithm_methods[0],
    #                 hyperparams=detection_algorithm_hyperparams[0])
    pipeline_description.add_step(step_5)
    counter += 1

    #Step 7: Predictions
    step_6 = build_step(arguments={'inputs': 'steps.' + str(counter) + '.produce', 'reference':    'steps.1.produce'},
               primitive_path='data_processing.construct_predictions')
    pipeline_description.add_step(step_6)
    counter+=1

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.' + str(counter) + '.produce')

    # print(pipeline_description.to_json_structure())
    return pipeline_description

def build_system_pipeline(config):
    """
    Build system pipline based on the user defined config
    Args:
        config:
            A user defined config specifying various primitives/hyperparams
            to be used for pipeline construction
    """

    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')
    
    # Step 0: Denormalise
    step_0 = build_step(primitive_path='common.denormalize',
                        arguments= {'inputs':'inputs.0'})
    pipeline_description.add_step(step_0)

    # Step 1: Convert the dataset to a DataFrame
    step_1 = build_step(primitive_path='data_processing.dataset_to_dataframe',
                        arguments={'inputs':'steps.0.produce'})
    pipeline_description.add_step(step_1)

    # Step 2: Read the csvs corresponding to the paths in the Dataframe in the form of arrays
    step_2 = build_step(primitive_path='common.csv_reader',
                        arguments={'inputs':'steps.1.produce'},
                        hyperparams={'use_columns':[0, 1],'return_result':'replace'})
    pipeline_description.add_step(step_2)

    # Step 3: Column Parser
    step_3 = build_step(primitive_path='data_processing.column_parser',
                        arguments={'inputs':'steps.2.produce'},
                        hyperparams={'parse_semantic_types':
                                    ['http://schema.org/Boolean','http://schema.org/Integer','http://schema.org/Float','https://metadata.datadrivendiscovery.org/types/FloatVector',]})
    pipeline_description.add_step(step_3)

    # Step 4: extract_columns_by_semantic_types(attributes)
    step_4 = build_step(primitive_path='data_processing.extract_columns_by_semantic_types',
                        arguments={'inputs': 'steps.3.produce'},
                        hyperparams={'semantic_types':
                                    ['https://metadata.datadrivendiscovery.org/types/Attribute']})
    pipeline_description.add_step(step_4)

    # Step 5: extract_columns_by_semantic_types(targets)
    step_5 = build_step(primitive_path='data_processing.extract_columns_by_semantic_types',
                        arguments={'inputs': 'steps.3.produce'},
                        hyperparams={'semantic_types':
                                    ['https://metadata.datadrivendiscovery.org/types/TrueTarget']})
    pipeline_description.add_step(step_5)

    attributes = 'steps.4.produce'
    targets = 'steps.5.produce'

    counter = 4
    flag = 1

    # Step 6: feature analysis
    feature_analysis_methods = get_primitive_list(config, 'feature_analysis')
    feature_analysis_hyperparams = get_primitives_hyperparam_list(config, 'feature_analysis')
    for i in range(len(feature_analysis_methods)):
        step_processing = build_step(arguments={'inputs': 'steps.' + str(counter) + '.produce'},
                                     primitive_path='feature_analysis.' + feature_analysis_methods[i],
                                     hyperparams=feature_analysis_hyperparams[i])
        pipeline_description.add_step(step_processing)
        counter += flag+1
        flag=0
        # if flag == 1:
        #     counter += 1
        #     flag = 0
        # counter += 1

    # Step 7: time_series processing
    timeseries_processing_methods = get_primitive_list(config, 'timeseries_processing')
    timeseries_processing_hyperparams = get_primitives_hyperparam_list(config, 'timeseries_processing')
    for i in range(len(timeseries_processing_methods)):

        step_processing = build_step(arguments={'inputs': 'steps.' + str(counter) + '.produce'},
                                     primitive_path='timeseries_processing.' + timeseries_processing_methods[i],
                                     hyperparams=timeseries_processing_hyperparams[i])
        pipeline_description.add_step(step_processing)

        if flag == 1:
            counter += 1
            flag = 0
        counter += 1

    # Step 8: detection algorithm
    detection_algorithm_methods = get_primitive_list(config,'detection_algorithm')
    detection_algorithm_hyperparams = get_primitives_hyperparam_list(config,'detection_algorithm')

    step_alg = build_step(primitive_path='detection_algorithm.'+detection_algorithm_methods[0],
                          arguments={'inputs': 'steps.' + str(counter) + '.produce'},
                          hyperparams=detection_algorithm_hyperparams[0])
    pipeline_description.add_step(step_alg)
    counter += 1

    # Step 9: Predictions
    step_8 = build_step(primitive_path='detection_algorithm.system_wise_detection',
                        arguments={'inputs':'steps.' + str(step_alg.index) + '.produce'},)
    pipeline_description.add_step(step_8)

    # Step 10: Output
    step_9 = build_step(primitive_path='data_processing.construct_predictions',
                        arguments={'inputs':'steps.' + str(step_8.index) + '.produce',
                                   'reference':'steps.1.produce'})
    pipeline_description.add_step(step_9)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.' + str(step_9.index) + '.produce')

    return pipeline_description


def sampling(args):
    from tensorflow.keras import backend as K

    from tensorflow.keras import backend as K
    z_mean, z_log = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))

    return z_mean + K.exp(0.5 * z_log) * epsilon

def find_save_folder():
    from pathlib import Path

    BASE_DIR = Path(__file__).parent.parent.absolute()
    TEMPLATES_DIR = BASE_DIR.joinpath('fitted_pipelines')
    return str(TEMPLATES_DIR) + '/'

def fit_pipeline(dataset, pipeline, metric='F1', seed=0):
    from axolotl.utils import schemas as schemas_utils
    from axolotl.backend.simple import SimpleRunner

    problem_description = generate_problem(dataset, metric)
    # print(problem_description.to_json_structure())
    backend = SimpleRunner(random_seed=seed)
    pipeline_result = backend.fit_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset])
    # print('='*50)
    # print(pipeline_result.fitted_pipeline_id)
    # print(backend.fitted_pipelines)
    fitted_pipeline = {
        'runtime': backend.fitted_pipelines[pipeline_result.fitted_pipeline_id],
        'dataset_metadata': dataset.metadata
    }

    return [fitted_pipeline,pipeline_result]

def save_fitted_pipeline(fitted_pipeline, save_path = find_save_folder()):
    import os
    import joblib

    runtime = fitted_pipeline['runtime']

    steps_state = runtime.steps_state

    pipeline_id = runtime.pipeline.id


    model_index = {}

    for i in range(len(steps_state)):

        if steps_state[i] != None and 'clf_' in runtime.steps_state[i]:
            model_name = type(runtime.steps_state[i]['clf_']).__name__
            model_index[str(model_name)] = i

            if 'AutoEncoder' in str(type(runtime.steps_state[i]['clf_'])) or 'VAE' in str(type(runtime.steps_state[i]['clf_'])) or 'LSTMOutlierDetector' in str(type(runtime.steps_state[i]['clf_'])) or 'DeeplogLstm' in str(type(runtime.steps_state[i]['clf_'])):
                runtime.steps_state[i]['clf_'].model_.save(save_path + str(pipeline_id) + '/model/' + str(model_name))
                runtime.steps_state[i]['clf_'].model_ = None
                joblib.dump(runtime.steps_state[i]['clf_'], save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

            elif 'SO_GAAL' in str(type(runtime.steps_state[i]['clf_'])):
                runtime.steps_state[i]['clf_'].combine_model.save(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_combine_model')
                runtime.steps_state[i]['clf_'].combine_model = None
                runtime.steps_state[i]['clf_'].discriminator.save(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
                runtime.steps_state[i]['clf_'].discriminator = None
                runtime.steps_state[i]['clf_'].generator.save(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_generator')
                runtime.steps_state[i]['clf_'].generator = None
                joblib.dump(runtime.steps_state[i]['clf_'], save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

            elif 'MO_GAAL' in str(type(runtime.steps_state[i]['clf_'])):
                runtime.steps_state[i]['clf_'].discriminator.save(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
                runtime.steps_state[i]['clf_'].discriminator = None
                joblib.dump(runtime.steps_state[i]['clf_'], save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

            elif 'Detector' in str(type(runtime.steps_state[i]['clf_'])):
                runtime.steps_state[i]['clf_']._model.model.save(save_path + str(pipeline_id) + '/model/' + str(model_name))
                runtime.steps_state[i]['clf_']._model.model = None
                joblib.dump(runtime.steps_state[i]['clf_'], save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

            else:
                if not os.path.isdir(save_path + str(pipeline_id) + '/'):
                    os.mkdir(save_path + str(pipeline_id) + '/')

    joblib.dump(fitted_pipeline, save_path + str(pipeline_id) + '/fitted_pipeline.pkl')
    joblib.dump(model_index, save_path + str(pipeline_id) + '/orders.pkl')

    return pipeline_id

def load_fitted_pipeline(pipeline_id, save_path = find_save_folder()):
    import joblib
    import keras

    orders = joblib.load(save_path + str(pipeline_id) + '/orders.pkl')

    fitted_pipeline = joblib.load(save_path + str(pipeline_id) + '/fitted_pipeline.pkl')

    for model_name, model_index in orders.items():
        if model_name == 'AutoEncoder':
            # print(model_name, model_index)
            # print(save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            # print(save_path + str(pipeline_id) + '/model/' + str(model_name))
            model = joblib.load(save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            model.model_ = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        elif model_name == 'VAE':
            model = joblib.load(save_path + str(pipeline_id) +  '/model/' + str(model_name) + '.pkl')

            model.model_ = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name), custom_objects = {'sampling': sampling})
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        elif model_name == 'SO_GAAL':
            model = joblib.load(save_path + str(pipeline_id) +  '/model/' + str(model_name) + '.pkl')
            model.discriminator = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
            model.combine_model = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_combine_model')
            model.generator = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_generator')
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        elif model_name == 'MO_GAAL':
            model = joblib.load(save_path + str(pipeline_id) +  '/model/' + str(model_name) + '.pkl')
            model.discriminator = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        elif model_name == 'LSTMOutlierDetector':
            model = joblib.load(save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            model.model_ = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        elif model_name == 'DeeplogLstm':
            model = joblib.load(save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            model.model_ = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        elif model_name == 'Detector':
            model = joblib.load(save_path + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            model._model.model = keras.models.load_model(save_path + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model

        else:
            fitted_pipeline = joblib.load(save_path + str(pipeline_id) + '/fitted_pipeline.pkl')

    return fitted_pipeline

def produce_fitted_pipeline(dataset, fitted_pipeline):
    from d3m.metadata import base as metadata_base
    from axolotl.backend.simple import SimpleRunner
    import uuid

    fitted_pipeline['dataset_metadata'] = dataset.metadata

    metadata_dict = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = fitted_pipeline['runtime']

    pipeline_result = backend.produce_pipeline(_id, [dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result

def fit_and_save_pipeline(dataset, pipeline, metric='F1', seed=0):
    fitted_pipeline = fit_pipeline(dataset, pipeline, 'F1_MACRO', 0)
    fitted_pipeline_id = save_fitted_pipeline(fitted_pipeline[0])
    return fitted_pipeline_id

def load_and_produce_pipeline(dataset, fitted_pipeline_id):
    fitted_pipeline = load_fitted_pipeline(fitted_pipeline_id)
    pipeline_result = produce_fitted_pipeline(dataset, fitted_pipeline)
    return pipeline_result

def get_primitive_list(config, module_name):
    """
    Get all the primitives of this module in user defined json

    Parameters
    ----------
    config: list
    module_name: str

    Returns
    -------
    primitive_list
    """
    module = config.get(module_name, default_primitive[module_name])
    # module = config.get(module_name, None)
    primitive_list = []
    
    if module is None:
        return primitive_list
    
    if module_name == 'detection_algorithm' and len(module) > 1:
        raise Exception("We currently only support one detection algorithm per pipeline.",
                        " Please modify the config")

    primitive_list = [module[i][0] for i in range(len(module))]
    return primitive_list


def get_primitives_hyperparam_list(config, module_name):
    """
    Get the hyperparams of all the primitives of this module in user defined json

    Parameters
    ----------
    config: list
    module_name: str

    Returns
    -------
    primitives_hyperparam_list
    """
    module = config.get(module_name, default_primitive[module_name])
    primitives_hyperparam_list = []
    
    if module is None:
        return primitives_hyperparam_list
    
    for i in range(len(module)):
        if len(module[i]) > 1:
            primitives_hyperparam_list.append(module[i][1])
        else:
            primitives_hyperparam_list.append(None)
    return primitives_hyperparam_list


def build_step(primitive_path,arguments=None, hyperparams=None):
    """
    Put the specified primitive and its parameters into the pipeline

    Parameters
    ----------
    primitive_path:
        A Python path under ``d3m.primitives`` namespace of a primitive.
    arguments: dict
        The arguments that need to be added to the primitive.
        key: argument name
        value: data_reference
    hyperparams: dict
        The hyperparams that need to be added to the primitive.
        key: hyperparam name
        value: data

    Returns
    -------
    step_processing:
        The primitive that has been initialized
    """
    #init primitive step
    step_processing = PrimitiveStep(
        primitive=index.get_primitive('d3m.primitives.tods.' + primitive_path))

    #add primitive arguments
    for key,value in arguments.items():
        step_processing.add_argument(name=key, argument_type=ArgumentType.CONTAINER, data_reference=value)

    # add hyperparams
    if hyperparams != None:
        for key in hyperparams:
            value = hyperparams[key]
            step_processing.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)

    # add output
    step_processing.add_output('produce')
    return step_processing

def get_evaluate_metric(y_true, y_pred, beta,metric):
    """

    Parameters
    ----------
    y_true: list[int]
        Ground truth of training dataset.
    y_pred: list[int]
        Prediction of training dataset.
    beta: float
        The parameter used to define F_beta score.
    metric:str
        The name of the metric to be yielded
    Returns
    -------

    """

    from sklearn.metrics import fbeta_score,f1_score,recall_score,precision_score,precision_recall_fscore_support
    # # TODO test specific algorithms
    # print('==========y_true============')
    # print(y_true)
    # print('==========y_pred============')
    # num=[0.0,1.0]
    # path = os.getcwd()
    # print(path)
    # df = pd.DataFrame(y_pred)
    # df.to_csv("../../../../mnt/tods/tods/data_test_"+search_space+".csv")
    # print(y_pred)
    t_pred = y_pred.copy()
    if isinstance(t_pred,DataFrame) or isinstance(t_pred,Series):
        t_pred.fillna(0,inplace = True)
    # print('='*50)
    # print(t_pred)
    
    
    precision_score = precision_score(y_true,t_pred)
    recall_score = recall_score(y_true,t_pred)
    f_beta = fbeta_score(y_true, t_pred, beta, average='macro')
    f1 = f1_score(y_true,t_pred,average='micro')
    f1_macro = f1_score(y_true,t_pred,average='macro')

    if metric == 'F1':
        performance_metrics = {'F1': f1}
    elif metric == 'F1_MACRO':
        performance_metrics = {'F1_MACRO': f1_macro}
    elif metric == 'RECALL':
        performance_metrics = {'RECALL':recall_score}
    elif metric == 'PRECISION':
        performance_metrics = {'PRECISION': precision_score}
    elif metric == 'F_BETA':
        performance_metrics = {'F_beta': f_beta}
    elif metric == 'ALL':
        performance_metrics = {'F_beta': f_beta, 'RECALL':recall_score,'PRECISION': precision_score, 'F1': f1,'F1_MACRO': f1_macro,}
    else:
        raise ValueError('The metric {} not supported.'.format(metric))

    return performance_metrics

def json_to_config(json):
    """
    Change the format of json to a config for building pipeline
    Parameters
    ----------
    json: dict
        Pipeline configuration defined by user

    Returns
    -------
    config: dict
        Convert the json into a form suitable for the parameters of build pipeline function
    """
    config = {}
    for key,value in json.items():
        primitive_list = []
        for primitive_name,primitive_value in value.items():
            if primitive_value:
                primitive_list.append([primitive_name,primitive_value])
            else:
                primitive_list.append([primitive_name])
        config[key] = primitive_list
    return config
