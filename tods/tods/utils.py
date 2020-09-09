
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

