
def generate_metrics():
    from d3m.metadata.problem import PerformanceMetric
    metrics = [{'metric': PerformanceMetric.F1, 'params': {'pos_label': '1'}},
              ]
    return metrics

def generate_data_preparation_params():
    from axolotl.utils import schemas as schemas_utils
    data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']
    return data_preparation_params
    
def generate_scoring_pipeline():
    from axolotl.utils import schemas as schemas_utils
    scoring_pipeline = schemas_utils.get_scoring_pipeline()
    return scoring_pipeline
    
def generate_data_preparation_pipeline():
    from axolotl.utils import schemas as schemas_utils
    data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
    return data_preparation_pipeline
    

def generate_dataset_problems(dataset_infos):
    """
    Args:
        dataset_infos: A list of dataset info, including `path` and `target`

    Returns:
        A list of Dataset and Problem
    """
    import pandas as pd
    from axolotl.utils import data_problem
    from d3m.metadata.problem import TaskKeyword, PerformanceMetric

    dataset_problems = []
    for dataset_info in dataset_infos:
        table_path = dataset_info['path']
        target = dataset_info['target']

        df = pd.read_csv(table_path)
        dataset, problem_description = data_problem.generate_dataset_problem(df,
                                                                             target_index=target,
                                                                             task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                             performance_metrics=[{'metric': PerformanceMetric.F1}])
        
        dataset_problems.append((dataset, problem_description))

    return dataset_problems
    
# FIXME: Currently only consider algorithm
def generate_pipelines(primitive_python_paths):
    """
    Args:
        primitive_python_paths: a list of primitive Python paths for algorithms
    
    Returns:
        the pipline description json
    """
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep
    from axolotl.utils import pipeline as pipeline_utils

    pipelines = []
    for primitive_python_path in primitive_python_paths:
        # Creating pipeline
        pipeline_description = Pipeline()
        pipeline_description.add_input(name='inputs')
        
        # The first three steps are fixed
        # Step 0: dataset_to_dataframe
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: column_parser
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        # Step 2: extract_columns_by_semantic_types(attributes)
        step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                                                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
        pipeline_description.add_step(step_2)

        # Step 3: extract_columns_by_semantic_types(targets)
        step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                                                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        pipeline_description.add_step(step_3)

        attributes = 'steps.2.produce'
        targets = 'steps.3.produce'

        # This one is what we want to test
        test_step = PrimitiveStep(primitive=index.get_primitive(primitive_python_path))
        test_step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
        test_step.add_output('produce')
        pipeline_description.add_step(test_step)

        # Finalize the pipeline
        final_step = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
        final_step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
        final_step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        final_step.add_output('produce')
        pipeline_description.add_step(final_step)

        pipeline_description.add_output(name='output predictions', data_reference='steps.5.produce')

        pipelines.append(pipeline_description)

    return pipelines
        
def test():
    # datasets to be tested
    dataset_infos = [
        {
            'path': 'datasets/anomaly/yahoo_sub_5/yahoo_sub_5_dataset/tables/learningData.csv',
            'target': 7
        },
        {
            'path': 'datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv',
            # 'path': 'datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv',
            'target': 3
        },
    ]

    # Algorithms to be tested
    # FIXME: Test more primitives
    primitive_python_paths = [
        'd3m.primitives.tods.detection_algorithm.pyod_ae',
        'd3m.primitives.tods.detection_algorithm.pyod_vae',
        'd3m.primitives.tods.detection_algorithm.pyod_cof',
        'd3m.primitives.tods.detection_algorithm.pyod_sod',
        'd3m.primitives.tods.detection_algorithm.pyod_abod',
        'd3m.primitives.tods.detection_algorithm.pyod_hbos',
        'd3m.primitives.tods.detection_algorithm.pyod_iforest',
        'd3m.primitives.tods.detection_algorithm.pyod_lof',
        'd3m.primitives.tods.detection_algorithm.pyod_knn',
        'd3m.primitives.tods.detection_algorithm.pyod_ocsvm',
        'd3m.primitives.tods.detection_algorithm.pyod_loda',
        # 'd3m.primitives.tods.detection_algorithm.pyod_cblof',
        'd3m.primitives.tods.detection_algorithm.pyod_sogaal',
        'd3m.primitives.tods.detection_algorithm.pyod_mogaal',
    ]

    dataset_problems = generate_dataset_problems(dataset_infos)
    pipelines = generate_pipelines(primitive_python_paths)
    metrics = generate_metrics()
    data_preparation_pipeline = generate_data_preparation_pipeline()
    scoring_pipeline = generate_scoring_pipeline()
    data_preparation_params = generate_data_preparation_params()

    # Start running
    from axolotl.backend.simple import SimpleRunner
    backend = SimpleRunner(random_seed=0)
    for i, dataset_problem in enumerate(dataset_problems):

        dataset, problem_description = dataset_problem
        for j, pipeline in enumerate(pipelines):

            print('Dataset:', i, 'Pipline:', j)

            pipeline_result = backend.evaluate_pipeline(problem_description=problem_description,
                                                        pipeline=pipeline,
                                                        input_data=[dataset],
                                                        metrics=metrics,
                                                        data_preparation_pipeline=data_preparation_pipeline,
                                                        scoring_pipeline=scoring_pipeline,
                                                        data_preparation_params=data_preparation_params)
            print('Results')
            print('----------------------------')
            print(pipeline_result)
            print('----------------------------')
            if pipeline_result.status == 'ERRORED':
                print('Scoring pipeline is {}'.format(scoring_pipeline.id))
                print('Data preparation pipeline is {}'.format(data_preparation_pipeline.id))
                raise ValueError('ERRORED for dataset {}, primitive {}'.format(dataset_infos[i], primitive_python_paths[j]))

if __name__ == "__main__":
    test()





