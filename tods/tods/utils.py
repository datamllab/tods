
def load_pipeline(pipeline_path):
    """Load a pipeline given a path

    Args:
        pipeline_path (str): The path to a pipeline file

    Returns:
        pipeline
    """
    from axolotl.utils import pipeline as pipeline_utils
    pipeline = pipeline_utils.load_pipeline(pipeline_path)

    return pipeline
    
def generate_dataset(df, target_index):
    """Generate dataset

    Args:
        df (pandas.DataFrame): dataset
        target_index (int): The column index of the target

    returns:
        dataset
    """
    from axolotl.utils import data_problem
    dataset = data_problem.import_input_data(df, target_index=target_index)

    return dataset

def generate_problem(dataset, metric):
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
    else:
        raise ValueError('The metric {} not supported.'.format(metric))

    problem_description = data_problem.generate_problem_description(dataset=dataset, 
                                                                    task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                    performance_metrics=performance_metrics)
    
    return problem_description

def evaluate_pipeline(dataset, pipeline, metric='F1', seed=0):
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


