import uuid
from d3m.metadata import base as metadata_base

from typing import List, Optional
from collections import OrderedDict
import uuid

import numpy as np

from d3m.container import DataFrame as d3m_dataframe
from d3m.primitive_interfaces.base import Hyperparams
from d3m.metadata import base as metadata_base

import argparse

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep


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


def construct_primitive_metadata(module, name, id, primitive_family, hyperparams =None, algorithm = None, flag_hyper = False, description = None):
    if algorithm == None:
        temp = [metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE]
    else:
        temp = []
        for alg in algorithm:
            temp.append(algorithm_type[alg])
    meta_dict = {
            "__author__ " : "DATA Lab @ Texas A&M University",
            'version': '0.3.0',
            'name': description,
            'python_path': 'd3m.primitives.tods.' + module + '.' + name,
            'source': {
                'name': "DATA Lab @ Texas A&M University",
                'contact': 'mailto:khlai037@tamu.edu',
            },
            'algorithm_types': temp,
            'primitive_family': primitive_families[primitive_family],#metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
            'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, id)),
            
        }
    #if name1 != None:
        #meta_dict['name'] = name1
    if hyperparams!=None:
        if flag_hyper == False:
            meta_dict['hyperparams_to_tune'] = hyperparams
        else:
            meta_dict['hyperparameters_to_tune'] = hyperparams
    metadata = metadata_base.PrimitiveMetadata(meta_dict,)
    return metadata
    
def load_pipeline(pipeline_path): # pragma: no cover
    """Load a pipeline given a path

    Args:
        pipeline_path (str): The path to a pipeline file

    Returns:
        pipeline
    """
    from axolotl.utils import pipeline as pipeline_utils
    pipeline = pipeline_utils.load_pipeline(pipeline_path)

    return pipeline
    
def generate_dataset(df, target_index, system_dir=None): # pragma: no cover
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

def generate_problem(dataset, metric): # pragma: no cover
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
    elif metric == 'ALL':
        performance_metrics = [{'metric': PerformanceMetric.PRECISION, 'params': {'pos_label': '1'}}, {'metric': PerformanceMetric.RECALL, 'params': {'pos_label': '1'}}, {'metric': PerformanceMetric.F1_MACRO, 'params': {}}, {'metric': PerformanceMetric.F1, 'params': {'pos_label': '1'}}]
    else:
        raise ValueError('The metric {} not supported.'.format(metric))

    problem_description = data_problem.generate_problem_description(dataset=dataset, 
                                                                    task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                    performance_metrics=performance_metrics)
    
    return problem_description

def evaluate_pipeline(dataset, pipeline, metric='F1', seed=0): # pragma: no cover
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
    print(dataset, pipeline, metric, data_preparation_pipeline)
    pipeline_result = backend.evaluate_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset],
                                                metrics=metrics,
                                                data_preparation_pipeline=data_preparation_pipeline,
                                                scoring_pipeline=scoring_pipeline,
                                                data_preparation_params=data_preparation_params)
    return pipeline_result





#Build pipeline

Inputs = d3m_dataframe
Outputs = d3m_dataframe


def build_pipeline(config):
    """Build a pipline based on the config
    """
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep
    
    algorithm = config.pop('algorithm', [('pyod_ae',None)])
    
    if len(algorithm)>1:
        raise Exception("We currently only support one detection algorithm per pipeline. Please modify the config")
    
    algorithm = algorithm[0]
    alg = algorithm[0]
    alg_hyper = algorithm[1]

    processing = config.pop('processing',[('statistical_maximum',)])
    processing_methods = [processing[i][0] for i in range(len(processing))]
    # Read augmentation hyperparameters
    processing_configs = []
    for i in range(len(processing)):
        if len(processing[i]) > 1:
            processing_configs.append(processing[i][1])
        else:
            processing_configs.append(None)
    
    #time-series processing
    timeseries_processing = config.pop('timeseries_processing',[])
    timeseries_processing_methods = [timeseries_processing[i][0] for i in range(len(timeseries_processing))]
    # Read time series processing hyperparameters
    timeseries_processing_configs = []
    for i in range(len(timeseries_processing)):
        if len(timeseries_processing[i]) > 1:
            timeseries_processing_configs.append(timeseries_processing[i][1])
        else:
            timeseries_processing_configs.append(None)
    
    #data_processing
    data_processing = config.pop('data_processing',[])
    data_processing_methods = [data_processing[i][0] for i in range(len(data_processing))]
    # Read data processing hyperparameters
    data_processing_configs = []
    for i in range(len(data_processing)):
        if len(data_processing[i]) > 1:
            data_processing_configs.append(data_processing[i][1])
        else:
            data_processing_configs.append(None)
            
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 0: dataset_to_dataframe
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: column_parser
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)
    
    
    #    #Step 4: data processing
#    for i in range(len(data_processing_methods)):
#
#        process_python_path = 'd3m.primitives.tods.data_processing.'+data_processing_methods[i]
#        step_processing = PrimitiveStep(primitive=index.get_primitive(process_python_path))
#        step_processing.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
#        if data_processing_configs[i] != None:
#            for key in data_processing_configs[i]:
#                value = data_processing_configs[i][key]
#                step_transformation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
#        step_processing.add_output('produce')
#        pipeline_description.add_step(step_processing)
#        if flag==1:
#            curr_step_no+=1
#            flag=0
#        curr_step_no+=1

    # Step 2: extract_columns_by_semantic_types(attributes)
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    pipeline_description.add_step(step_2)

    # Step 3: extract_columns_by_semantic_types(targets)
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    pipeline_description.add_step(step_3)

    attributes = 'steps.2.produce'
    targets = 'steps.3.produce'
    
        
    curr_step_no = 2
    flag = 1

    # Step 4: processing
    for i in range(len(processing_methods)):
    
        process_python_path = 'd3m.primitives.tods.feature_analysis.'+processing_methods[i]
        step_processing = PrimitiveStep(primitive=index.get_primitive(process_python_path))
        step_processing.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        if processing_configs[i] != None:
            for key in processing_configs[i]:
                value = processing_configs[i][key]
                step_transformation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_processing.add_output('produce')
        pipeline_description.add_step(step_processing)
        if flag==1:
            curr_step_no+=1
            flag=0
        curr_step_no+=1
     
         
    # Step 4: processing
    for i in range(len(timeseries_processing_methods)):
    
        process_python_path = 'd3m.primitives.tods.timeseries_processing.'+timeseries_processing_methods[i]
        step_processing = PrimitiveStep(primitive=index.get_primitive(process_python_path))
        step_processing.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        if timeseries_processing_configs[i] != None:
            for key in timeseries_processing_configs[i]:
                value = timeseries_processing_configs[i][key]
                step_transformation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_processing.add_output('produce')
        pipeline_description.add_step(step_processing)
        if flag==1:
            curr_step_no+=1
            flag=0
        curr_step_no+=1
        
    
    
    
    # Step 5: algorithm
    alg_python_path = 'd3m.primitives.tods.detection_algorithm.' + alg
    
    step_5 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
    if alg_hyper!=None:
        for key, value in alg_hyper.items():
            step_5.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)
    curr_step_no+=1

    # Step 6: Predictions
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=f'steps.{step_5.index}.produce')
    step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)
    curr_step_no+=1

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference=f'steps.{step_6.index}.produce')


#    print(pipeline_description)
    # Output to json
    data = pipeline_description.to_json()
    print("----------------------BUILD PIPELINE DATA: ", data)
#    with open('autoencoder_pipeline.json', 'w') as f:
#        f.write(data)
#        print(data)
    return pipeline_description




#Build System Pipeline
def build_system_pipeline(config):


    algorithm = config.pop('algorithm', [('pyod_ae',None)])
    
    if len(algorithm)>1:
        raise Exception("We currently only support one detection algorithm per pipeline. Please modify the config")
    
    algorithm = algorithm[0]
    alg = algorithm[0]
    if len(algorithm)==1:
        alg_hyper={}
    else: alg_hyper = algorithm[1]

    processing = config.pop('processing',[('statistical_maximum',)])
    processing_methods = [processing[i][0] for i in range(len(processing))]
    # Read augmentation hyperparameters
    processing_configs = []
    for i in range(len(processing)):
        if len(processing[i]) > 1:
            processing_configs.append(processing[i][1])
        else:
            processing_configs.append(None)
    #time-series processing
    timeseries_processing = config.pop('timeseries_processing',[])
    timeseries_processing_methods = [timeseries_processing[i][0] for i in range(len(timeseries_processing))]
    # Read time series processing hyperparameters
    timeseries_processing_configs = []
    for i in range(len(timeseries_processing)):
        if len(timeseries_processing[i]) > 1:
            timeseries_processing_configs.append(timeseries_processing[i][1])
        else:
            timeseries_processing_configs.append(None)
    
    #data_processing
    data_processing = config.pop('data_processing',[])
    data_processing_methods = [data_processing[i][0] for i in range(len(data_processing))]
    # Read data processing hyperparameters
    data_processing_configs = []
    for i in range(len(data_processing)):
        if len(data_processing[i]) > 1:
            data_processing_configs.append(data_processing[i][1])
        else:
            data_processing_configs.append(None)
            
            
            
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    #Step 0: Denormalise
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.common.denormalize'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    #Step 1: Convert the dataset to a DataFrame
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    #Step 2: Read the csvs corresponding to the paths in the Dataframe in the form of arrays
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.common.csv_reader'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(name = 'use_columns', argument_type=ArgumentType.VALUE, data = [0,1])
    step_2.add_hyperparameter(name = 'return_result', argument_type=ArgumentType.VALUE, data = 'replace')
    pipeline_description.add_step(step_2)

    #Step 3: Column Parser
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='parse_semantic_types', argument_type=ArgumentType.VALUE,
                                      data=['http://schema.org/Boolean','http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector',])
    pipeline_description.add_step(step_3)

    # Step 4: extract_columns_by_semantic_types(attributes)
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_output('produce')
    step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    pipeline_description.add_step(step_4)

    # Step 5: extract_columns_by_semantic_types(targets)
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_5.add_output('produce')
    step_5.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    pipeline_description.add_step(step_5)

    attributes = 'steps.4.produce'
    targets = 'steps.5.produce'
    
    
    curr_step_no = 4
    flag = 1

    # Step 6: processing
    for i in range(len(processing_methods)):
    
        process_python_path = 'd3m.primitives.tods.feature_analysis.'+processing_methods[i]
        step_processing = PrimitiveStep(primitive=index.get_primitive(process_python_path))
        step_processing.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        if processing_configs[i] != None:
            for key in processing_configs[i]:
                value = processing_configs[i][key]
                step_transformation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_processing.add_output('produce')
        pipeline_description.add_step(step_processing)
        if flag==1:
            curr_step_no+=1
            flag=0
        curr_step_no+=1
     
         
    # Step 6: processing
    for i in range(len(timeseries_processing_methods)):

        process_python_path = 'd3m.primitives.tods.timeseries_processing.transformation.'+timeseries_processing_methods[i]
        step_processing = PrimitiveStep(primitive=index.get_primitive(process_python_path))
        step_processing.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
        if timeseries_processing_configs[i] != None:
            for key in timeseries_processing_configs[i]:
                value = timeseries_processing_configs[i][key]
                step_transformation.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
        step_processing.add_output('produce')
        pipeline_description.add_step(step_processing)
        if flag==1:
            curr_step_no+=1
            flag=0
        curr_step_no+=1

    
    
    
    # Step 7: algorithm
    alg_python_path = 'd3m.primitives.tods.detection_algorithm.' + alg

    step_alg = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
    step_alg.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(curr_step_no)+'.produce')
    if alg_hyper!=None:
        for key, value in alg_hyper.items():
            step_alg.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
    step_alg.add_output('produce_score')
    pipeline_description.add_step(step_alg)
    curr_step_no+=1


    # Step 8: Predictions
    #step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
    step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.system_wise_detection'))
    step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(step_alg.index)+'.produce_score')
    #step_8.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_8.add_output('produce')
    pipeline_description.add_step(step_8)

    step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
    step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.'+str(step_8.index)+'.produce')
    step_9.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_9.add_output('produce')
    pipeline_description.add_step(step_9)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.'+str(step_9.index)+'.produce')
    
    


#    print(pipeline_description)
    # Output to json
    data = pipeline_description.to_json()
    print("----------------------BUILD System PIPELINE DATA: ", data)
#    with open('autoencoder_pipeline.json', 'w') as f:
#        f.write(data)
#        print(data)
    return pipeline_description

