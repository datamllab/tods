
def generate_dataset_problem(df, target_index, metric):
    """
    A wrapper for generating dataset and problem

    Args:
        df (pandas.DataFrame): dataset
        target_index (int): The column index of the target
        metric (str): `F1` for computing F1 on label 1, 'F1_MACRO` for 
            macro-F1 on both 0 and 1

    returns:
        dataset, problem
    """
    from axolotl.utils import data_problem
    from d3m.metadata.problem import TaskKeyword, PerformanceMetric

    if metric == 'F1':
        performance_metrics = [{'metric': PerformanceMetric.F1, 'params': {'pos_label': '1'}}]
    elif metric == 'F1_MACRO':
        performance_metrics = [{'metric': PerformanceMetric.F1_MACRO, 'params': {}}]
    else:
        raise ValueError('The metric {} not supported.'.format(metric))
        
        
    dataset, problem_description = data_problem.generate_dataset_problem(df,
                                                                         target_index=target_index,
                                                                         task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                         performance_metrics=performance_metrics)

    return dataset, problem_description

def evaluate_pipeline(problem_description, dataset, pipeline):
    from axolotl.utils import schemas as schemas_utils
    from axolotl.backend.simple import SimpleRunner
    data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
    scoring_pipeline = schemas_utils.get_scoring_pipeline()
    data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']
    metrics = problem_description['problem']['performance_metrics']

    backend = SimpleRunner(random_seed=0) 
    pipeline_result = backend.evaluate_pipeline(problem_description=problem_description,
                                                pipeline=pipeline,
                                                input_data=[dataset],
                                                metrics=metrics,
                                                data_preparation_pipeline=data_preparation_pipeline,
                                                scoring_pipeline=scoring_pipeline,
                                                data_preparation_params=data_preparation_params)
    try:
        for error in pipeline_result.error:
            if error is not None:
                raise error
    except:
        import traceback
        traceback.print_exc()

    return pipeline_result


