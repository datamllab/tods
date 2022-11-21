import uuid
from d3m.metadata import base as metadata_base

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


