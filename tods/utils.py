
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
    pipeline_result = backend.evaluate_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset],
                                                metrics=metrics,
                                                data_preparation_pipeline=data_preparation_pipeline,
                                                scoring_pipeline=scoring_pipeline,
                                                data_preparation_params=data_preparation_params)
    return pipeline_result


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def sampling(args):
    from tensorflow.keras import backend as K
    """Reparametrisation by sampling from Gaussian, N(0,I)
    To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
    with latent variables z: z = z_mean + sqrt(var) * epsilon
    Parameters
    ----------
    args : tensor
        Mean and log of variance of Q(z|X).

    Returns
    -------
    z : tensor
        Sampled latent variable.
    """
    from tensorflow.keras import backend as K
    z_mean, z_log = args
    batch = K.shape(z_mean)[0]  # batch size
    dim = K.int_shape(z_mean)[1]  # latent dimension
    epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0

    return z_mean + K.exp(0.5 * z_log) * epsilon


def save(dataset, pipeline, metric='F1', seed=0):
    from axolotl.utils import schemas as schemas_utils
    from axolotl.backend.simple import SimpleRunner
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    import os

    # from dill import dumps, loads

    problem_description = generate_problem(dataset, metric)

    backend = SimpleRunner(random_seed=seed) 
    pipeline_result = backend.fit_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset])

    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error

    fitted_pipeline = {
        'runtime': backend.fitted_pipelines[pipeline_result.fitted_pipeline_id],
        'dataset_metadata': dataset.metadata
    }


    steps_state = backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state
    
    pipeline_id = backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].pipeline.id

    model_index = -1

    for i in range(len(steps_state)):
        if steps_state[i] != None:
            model_index = i
            # backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/keras_model')
            # backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_ = None
            # joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')

            # steps_state[i]['clf_'] = 'place_holder'

    print(steps_state)

    model_name = type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_']).__name__
    print(model_name)

    if 'AutoEncoder' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'])):
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_ = None
        joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')

        steps_state[model_index]['clf_'] = 'place_holder'

        joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    elif 'VAE' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'])):
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_ = None
        joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')

        steps_state[model_index]['clf_'] = 'place_holder'

        joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')

        # path = 'fitted_pipelines/' + str(pipeline_id)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_.save('fitted_pipelines')
        # tf.keras.models.save_model(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines')


        # joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')

        # path = 'fitted_pipelines/' + str(pipeline_id) + '/model'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # with open('fitted_pipelines/' + str(pipeline_id) + '/model/model.json', 'w') as json_file:
        #     json_file.write(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].to_json())
    elif 'LSTMOutlierDetector' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'])):
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_ = None
        joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')

        steps_state[model_index]['clf_'] = 'place_holder'

        joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    elif 'DeeplogLstm' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'])):
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'].model_ = None
        joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')

        steps_state[model_index]['clf_'] = 'place_holder'

        joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    elif 'Detector' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'])):
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_']._model.model.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
        backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_']._model.model = None
        joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')

        steps_state[model_index]['clf_'] = 'place_holder'

        joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')

    

    else:
        print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[model_index]['clf_'])
        if not os.path.isdir('fitted_pipelines/' + str(pipeline_id) + '/'):
            os.mkdir('fitted_pipelines/' + str(pipeline_id) + '/')
        joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')


    return pipeline_id


def load(dataset, pipeline_id):
    from d3m.metadata import base as metadata_base
    from axolotl.backend.simple import SimpleRunner
    import uuid
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    import os

    from d3m.runtime import Runtime

    path = 'fitted_pipelines/' + str(pipeline_id) + '/model'

    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    print(model_list)

    if model_list[0] == 'VAE':
        fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
        model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
        model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]), custom_objects={'sampling': sampling})
    elif model_list[0] == 'AutoEncoder':
        fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
        model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
        model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))
    elif model_list[0] == 'LSTMOutlierDetector':
        fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
        model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
        model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))
    elif model_list[0] == 'DeeplogLstm':
        fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
        model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
        model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))
    elif  model_list[0] == 'Detector':
        fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
        model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
        model._model.model = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))
    else:
        fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')


    steps_state = fitted_pipeline['runtime'].steps_state

    for i in range(len(steps_state)):
        if steps_state[i] != None:
            if steps_state[i]['clf_'] == 'place_holder':
                steps_state[i]['clf_'] = model


    dataset.metadata = fitted_pipeline['dataset_metadata']

    metadata_dict = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    # metadata_dict['location_base_uris'] = [pathlib.Path(os.path.abspath(test_media_dir)).as_uri()+'/']
    dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = fitted_pipeline['runtime']

    # Produce
    pipeline_result = backend.produce_pipeline(_id, [dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result

# ---------------------------------------------------------------------------------------------------------------------------------------------------

def fit_pipeline(dataset, pipeline, metric='F1', seed=0): # pragma: no cover

    from axolotl.utils import schemas as schemas_utils
    from axolotl.backend.simple import SimpleRunner
    # import pickle
    import joblib
    import json
    import jsonpickle
    import gzip
    import ubjson
    import tensorflow as tf
    from tensorflow import keras
    import ruamel.yaml
    # from dill import dumps, loads

    problem_description = generate_problem(dataset, metric)

    backend = SimpleRunner(random_seed=seed) 
    pipeline_result = backend.fit_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset])

    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error

    fitted_pipeline = {
        'runtime': backend.fitted_pipelines[pipeline_result.fitted_pipeline_id],
        'dataset_metadata': dataset.metadata
    }

    temp = Resultt(fitted_pipeline['runtime'], fitted_pipeline['dataset_metadata'])

    # frozen = jsonpickle.encode(fitted_pipeline)

    # with gzip.open('testfitted', 'wb') as f:
    #     ubjson.dump(fitted_pipeline, f)

    # tf.keras.models.save_model(backend, '/fitted')

    # import torch
    # torch.save(fitted_pipeline, 'fitted_pipe')

    # out = open('id.pkl', 'wb', pickle.HIGHEST_PROTOCOL)
    # pickle.dump(pipeline_result.fitted_pipeline_id, out)
    # out.close()

    # out = open('metadata.pkl', 'wb', pickle.HIGHEST_PROTOCOL)
    # pickle.dump(dataset.metadata, out)
    # out.close()

    # with open("temp.json", "w") as outfile:
    #     json.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].__dict__, outfile)

    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].pipeline)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].hyperparams)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].problem_description)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].context)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].random_seed)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].volumes_dir)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].scratch_dir)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].is_standard_pipeline)
    # print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].users)

    print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5])

    print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'])

    print(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'].model_.save('fitted')))

    backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'].model_ = None
    joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'], 'temp.pkl')

    model = joblib.load('temp.pkl')

    model.model_ = keras.models.load_model('fitted')

    print(model)




    # backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'].save('saved_model/my_model')
    # dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'], 'clf.joblib')

    # import dill as pickle
    # with open('fitted.pkl', 'wb') as file:
    #     pickle.dump(fitted_pipeline, file)

    # with open("fitted.pkl", 'wb') as outp:
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].pipeline, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].hyperparams, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].problem_description, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].context, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].random_seed, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].volumes_dir, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].scratch_dir, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].is_standard_pipeline, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].users, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state, outp, pickle.HIGHEST_PROTOCOL)
    # #     pickle.dump(dataset.metadata, outp, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[5]['clf_'], outp, pickle.HIGHEST_PROTOCOL)

    # # yaml = ruamel.yaml.YAML()
    # # yaml.register_class(Resultt)
    # # yaml.dump([Resultt(fitted_pipeline['runtime'], fitted_pipeline['dataset_metadata'])], sys.stdout)

    # yaml = ruamel.yaml.YAML(typ='unsafe')
    # yaml.dump(fitted_pipeline, sys.stdout)

    # with open("sample.json", "w") as outfile:
    #     json.dump(temp.toJSON(), outfile)

    # joblib.dump(fitted_pipeline, 'test')

    return pipeline_result, pipeline_result.output, fitted_pipeline

def load_fitted_pipeline_by_pipeline(dataset, fitted_pipeline):
    from d3m.metadata import base as metadata_base
    from axolotl.backend.simple import SimpleRunner
    import uuid

    dataset.metadata = fitted_pipeline['dataset_metadata']

    metadata_dict = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    # metadata_dict['location_base_uris'] = [pathlib.Path(os.path.abspath(test_media_dir)).as_uri()+'/']
    dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = fitted_pipeline['runtime']

    # Produce
    pipeline_result = backend.produce_pipeline(_id, [dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result

def load_fitted_pipeline_by_id(dataset):
    from d3m.metadata import base as metadata_base
    from axolotl.backend.simple import SimpleRunner
    import uuid
    import pickle

    from d3m.runtime import Runtime

    # infile = open("id.pkl",'rb')
    # idd = pickle.load(infile)
    # infile.close()

    # infile = open("metadata.pkl",'rb')
    # metadata = pickle.load(infile)
    # infile.close()

    with open("fitted.pkl", "rb") as f:
        pipeline = pickle.load(f)
        hyperparams = pickle.load(f)
        problem_description = pickle.load(f)
        context = pickle.load(f)
        random_seed = pickle.load(f)
        volumes_dir = pickle.load(f)
        scratch_dir = pickle.load(f)
        is_standard_pipeline = pickle.load(f)
        users = pickle.load(f)
        steps_state = pickle.load(f)
        metadata = pickle.load(f)


    # print(pipeline)
    # print(hyperparams)
    # print(problem_description)
    # print(context)
    # print(random_seed)
    # print(volumes_dir)
    # print(scratch_dir)
    # print(is_standard_pipeline)
    # print(users)


    # problem_description = pickle.load(open('fitted.pkl', 'rb'))
    # print(problem_description)

    # context = pickle.load(open('fitted.pkl', 'rb'))
    # print(context)

    # random_seed = pickle.load(open('fitted.pkl', 'rb'))
    # print(random_seed)

    # volumes_dir = pickle.load(open('fitted.pkl', 'rb'))
    # print(volumes_dir)

    # scratch_dir = pickle.load(open('fitted.pkl', 'rb'))
    # print(scratch_dir)

    # is_standard_pipeline = pickle.load(open('fitted.pkl', 'rb'))
    # print(is_standard_pipeline)

    # users = pickle.load(open('fitted.pkl', 'rb'))
    # print(users)

    new = Runtime(pipeline = pipeline,
    hyperparams = hyperparams,
    problem_description = problem_description,
    context = context,
    random_seed = random_seed,
    volumes_dir = volumes_dir,
    scratch_dir = scratch_dir,
    is_standard_pipeline = is_standard_pipeline,
    users = users)

    new.steps_state = steps_state


    dataset.metadata = metadata

    metadata_dict = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    # metadata_dict['location_base_uris'] = [pathlib.Path(os.path.abspath(test_media_dir)).as_uri()+'/']
    dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = new

    # Produce
    pipeline_result = backend.produce_pipeline(_id, [dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result


def save2(dataset, pipeline, metric='F1', seed=0):
    from axolotl.utils import schemas as schemas_utils
    from axolotl.backend.simple import SimpleRunner
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    import os

    # import dill as pickle

    # from dill import dumps, loads

    problem_description = generate_problem(dataset, metric)

    backend = SimpleRunner(random_seed=seed) 
    pipeline_result = backend.fit_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset])

    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error

    fitted_pipeline = {
        'runtime': backend.fitted_pipelines[pipeline_result.fitted_pipeline_id],
        'dataset_metadata': dataset.metadata
    }

    original_fitted_pipeline = fitted_pipeline

    print(fitted_pipeline['runtime'].steps_state)

    steps_state = backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state

    pipeline_id = backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].pipeline.id

    model_index = {}

    for i in range(len(steps_state)):
        if steps_state[i] != None:
            model_name = type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_']).__name__
            model_index[str(model_name)] = i

            if 'AutoEncoder' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                print('---------------------------------------------------------------------------------------------------------')
                print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_)
                print('---------------------------------------------------------------------------------------------------------')
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_ = None
                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

            elif 'VAE' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_ = None
                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

            elif 'SO_GAAL' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].combine_model.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_combine_model')
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].combine_model = None

                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].discriminator.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].discriminator = None

                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].generator.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_generator')
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].generator = None


                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

            elif 'MO_GAAL' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                print(vars(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_']))

                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].discriminator.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].discriminator = None

                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

            elif 'LSTMOutlierDetector' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_ = None
                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

                joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
            elif 'DeeplogLstm' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'].model_ = None
                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

                joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
            elif 'Detector' in str(type(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])):
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_']._model.model.save('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
                backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_']._model.model = None
                joblib.dump(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'], 'fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')

                # steps_state[i]['clf_'] = 'place_holder'

                joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
            
            else:
                print(backend.fitted_pipelines[pipeline_result.fitted_pipeline_id].steps_state[i]['clf_'])
                if not os.path.isdir('fitted_pipelines/' + str(pipeline_id) + '/'):
                    os.mkdir('fitted_pipelines/' + str(pipeline_id) + '/')
                joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')


    
    joblib.dump(fitted_pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    joblib.dump(model_index, 'fitted_pipelines/' + str(pipeline_id) + '/orders.pkl')

    # joblib.dump(pipeline, 'fitted_pipelines/' + str(pipeline_id) + '/original_description.pkl'))

    return pipeline_id, original_fitted_pipeline


def load2(dataset, pipeline_id):
    from d3m.metadata import base as metadata_base
    from axolotl.backend.simple import SimpleRunner
    import uuid
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    import os

    from d3m.runtime import Runtime


    orders = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/orders.pkl')
    print('orders', orders)

    fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')

    for model_name, model_index in orders.items():
        print(model_name, model_index)
        if model_name == 'AutoEncoder':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            print(model)
            model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model
        elif model_name == 'VAE':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) +  '/model/' + str(model_name) + '.pkl')
            print(model)
            model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name), custom_objects = {'sampling': sampling})
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model
        elif model_name == 'SO_GAAL':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) +  '/model/' + str(model_name) + '.pkl')
            print(model)
            
            model.discriminator = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')
            model.combine_model = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_combine_model')
            model.generator = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_generator')
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model
        elif model_name == 'MO_GAAL':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) +  '/model/' + str(model_name) + '.pkl')
            model.discriminator = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '_discriminator')

            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model
        elif model_name == 'LSTMOutlierDetector':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            print(model)
            model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model
        elif model_name == 'DeeplogLstm':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            print(model)
            model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model
        elif model_name == 'Detector':
            model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name) + '.pkl')
            print(model)
            model._model.model = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_name))
            fitted_pipeline['runtime'].steps_state[model_index]['clf_'] = model



            # model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
            # model._model.model = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))
        else:
            fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')


    print(fitted_pipeline['runtime'].steps_state)






    # path = 'fitted_pipelines/' + str(pipeline_id) + '/model'

    # model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    # print(model_list)

    # if model_list[0] == 'VAE':
    #     fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    #     model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
    #     model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]), custom_objects={'sampling': sampling})
    # elif model_list[0] == 'AutoEncoder':
    #     fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    #     model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
    #     model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))

    # steps_state = fitted_pipeline['runtime'].steps_state

    # for i in range(len(steps_state)):
    #     if steps_state[i] != None:
    #         if steps_state[i]['clf_'] == 'place_holder':
    #             steps_state[i]['clf_'] = model


    dataset.metadata = fitted_pipeline['dataset_metadata']

    metadata_dict = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    # metadata_dict['location_base_uris'] = [pathlib.Path(os.path.abspath(test_media_dir)).as_uri()+'/']
    dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = fitted_pipeline['runtime']

    print(fitted_pipeline['runtime'].pipeline.description)

    # Produce
    pipeline_result = backend.produce_pipeline(_id, [dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result, fitted_pipeline


def testss(dataset, pipeline_id):
    from d3m.metadata import base as metadata_base
    from axolotl.backend.simple import SimpleRunner
    import uuid
    import joblib
    import tensorflow as tf
    from tensorflow import keras
    import os

    from d3m.runtime import Runtime

    # path = 'fitted_pipelines/' + str(pipeline_id) + '/model'

    # model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    # print(model_list)

    # if model_list[0] == 'VAE':
    #     fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    #     model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
    #     model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]), custom_objects={'sampling': sampling})
    # elif model_list[0] == 'AutoEncoder':
    #     fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')
    #     model = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/model/model.pkl')
    #     model.model_ = keras.models.load_model('fitted_pipelines/' + str(pipeline_id) + '/model/' + str(model_list[0]))
    # else:
    fitted_pipeline = joblib.load('fitted_pipelines/' + str(pipeline_id) + '/fitted_pipeline.pkl')


    steps_state = fitted_pipeline['runtime'].steps_state

    for i in range(len(steps_state)):
        if steps_state[i] != None:
            if steps_state[i]['clf_'] == 'place_holder':
                steps_state[i]['clf_'] = model


    dataset.metadata = fitted_pipeline['dataset_metadata']

    metadata_dict = dataset.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))
    metadata_dict = {key: metadata_dict[key] for key in metadata_dict}
    # metadata_dict['location_base_uris'] = [pathlib.Path(os.path.abspath(test_media_dir)).as_uri()+'/']
    dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), metadata_dict)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    _id = str(uuid.uuid4())
    backend.fitted_pipelines[_id] = fitted_pipeline['runtime']

    # Produce
    pipeline_result = backend.produce_pipeline(_id, [dataset])
    if pipeline_result.status == "ERRORED":
        raise pipeline_result.error
    return pipeline_result


def check_runtime_diff(saved_runtime, loaded_runtime):
    #first check attributes of pipeline
    print(saved_runtime['runtime'].__dict__)
    print(loaded_runtime['runtime'].__dict__)
    #then check steps state
    print(saved_runtime['runtime'].steps_state)
    print(loaded_runtime['runtime'].steps_state)
    #then check clf
    model_indexes = []
    for i in range(len(saved_runtime['runtime'].steps_state)):  
        if saved_runtime['runtime'].steps_state[i] != None:
            model_indexes.append(i)

    for i in model_indexes:
        print(saved_runtime['runtime'].steps_state[i]['clf_'])
        print(loaded_runtime['runtime'].steps_state[i]['clf_'])
    #then comapre model based on model
    for i in model_indexes:
        print(saved_runtime['runtime'].steps_state[i]['clf_'].model_)
        print(loaded_runtime['runtime'].steps_state[i]['clf_'].model_)