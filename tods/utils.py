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
    print("data preparation pipeline == ", data_preparation_pipeline)
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

def sampling(args):
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

    backend = SimpleRunner(random_seed=seed) 
    pipeline_result = backend.fit_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset])

    fitted_pipeline = {
        'runtime': backend.fitted_pipelines[pipeline_result.fitted_pipeline_id],
        'dataset_metadata': dataset.metadata
    }

    return fitted_pipeline

def save_fitted_pipeline(fitted_pipeline, save_path = find_save_folder()):
    import os
    import joblib

    runtime = fitted_pipeline['runtime']

    steps_state = runtime.steps_state

    pipeline_id = runtime.pipeline.id

    model_index = {}

    for i in range(len(steps_state)):
        if steps_state[i] != None:
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

    dataset.metadata = fitted_pipeline['dataset_metadata']

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
    fitted_pipeline_id = save_fitted_pipeline(fitted_pipeline)
    return fitted_pipeline_id

def load_and_produce_pipeline(dataset, fitted_pipeline_id):
    fitted_pipeline = load_fitted_pipeline(fitted_pipeline_id)
    pipeline_result = produce_fitted_pipeline(dataset, fitted_pipeline)
    return pipeline_result
