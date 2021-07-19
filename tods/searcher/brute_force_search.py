# A Brute-Force Search
import uuid
import random

from d3m.metadata.pipeline import Pipeline

from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import  schemas as schemas_utils

class BruteForceSearch(PipelineSearchBase): # pragma: no cover
    def __init__(self, problem_description, backend, *, primitives_blocklist=None, ranking_function=None):
        super().__init__(problem_description=problem_description, backend=backend,
                primitives_blocklist=primitives_blocklist, ranking_function=ranking_function)
        if self.ranking_function is None:
            self.ranking_function = _rank_first_metric

        # Find the candidates
        self.task_description = schemas_utils.get_task_description(self.problem_description['problem']['task_keywords'])
        self.available_pipelines = self._return_pipelines(
                            self.task_description['task_type'], self.task_description['task_subtype'], self.task_description['data_types'])
        
        self.metrics = self.problem_description['problem']['performance_metrics']
        self.data_preparation_pipeline = _generate_data_preparation_pipeline()
        self.scoring_pipeline = _generate_scoring_pipeline()
        self.data_preparation_params = _generate_data_preparation_params()

        self.current_pipeline_index = 0
        self.offset = 1

    def evaluate(self, pipeline_to_eval, input_data=None):
        if input_data is None:
            input_data = self.input_data
        pipeline_result = self.backend.evaluate_pipeline(
                problem_description=self.problem_description,
                pipeline=pipeline_to_eval,
                input_data=input_data,
                metrics=self.metrics,
                data_preparation_pipeline=self.data_preparation_pipeline,
                scoring_pipeline=self.scoring_pipeline,
                data_preparation_params=self.data_preparation_params)
        
        return pipeline_result

    def _search(self, time_left):
        # Read all the pipelines to be evaluated
        pipelines_to_eval = self.available_pipelines[self.current_pipeline_index: self.current_pipeline_index+self.offset]
        self.current_pipeline_index += 1
        
        pipeline_results = self.backend.evaluate_pipelines(
                problem_description=self.problem_description,
                pipelines=pipelines_to_eval,
                input_data=self.input_data,
                metrics=self.metrics,
                data_preparation_pipeline=self.data_preparation_pipeline,
                scoring_pipeline=self.scoring_pipeline,
                data_preparation_params=self.data_preparation_params)

        # DEBUG
        ####################
        for pipeline_result in pipeline_results:
            try:
                for error in pipeline_result.error:
                    if error is not None:
                        raise error
            except:
                import traceback
                traceback.print_exc()
        ####################

        return [self.ranking_function(pipeline_result) for pipeline_result in pipeline_results]

    def _return_pipelines(self, task_type, task_subtype, data_type):
        pipeline_candidates = _generate_pipelines(primitive_python_paths)
        return pipeline_candidates

primitive_python_paths = { # pragma: no cover
    'data_processing': [
        #'d3m.primitives.tods.data_processing.time_interval_transform',
        #'d3m.primitives.tods.data_processing.categorical_to_binary',
        'd3m.primitives.tods.data_processing.column_filter',
        #'d3m.primitives.tods.data_processing.timestamp_validation',
        #'d3m.primitives.tods.data_processing.duplication_validation',
        #'d3m.primitives.tods.data_processing.continuity_validation',
    ],
    'timeseries_processing': [
        'd3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler',
        # 'd3m.primitives.tods.timeseries_processing.transformation.standard_scaler',
        # 'd3m.primitives.tods.timeseries_processing.transformation.power_transformer',
        # 'd3m.primitives.tods.timeseries_processing.transformation.quantile_transformer',
        # 'd3m.primitives.tods.timeseries_processing.transformation.moving_average_transform',
        # 'd3m.primitives.tods.timeseries_processing.transformation.simple_exponential_smoothing',
        #'d3m.primitives.tods.timeseries_processing.transformation.holt_smoothing',
        #'d3m.primitives.tods.timeseries_processing.transformation.holt_winters_exponential_smoothing',
        #'d3m.primitives.tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition',
    ],
    'feature_analysis': [
        #'d3m.primitives.tods.feature_analysis.auto_correlation',
        'd3m.primitives.tods.feature_analysis.statistical_mean',
        # 'd3m.primitives.tods.feature_analysis.statistical_median',
        # 'd3m.primitives.tods.feature_analysis.statistical_g_mean',
        # 'd3m.primitives.tods.feature_analysis.statistical_abs_energy',
        # 'd3m.primitives.tods.feature_analysis.statistical_abs_sum',
        # 'd3m.primitives.tods.feature_analysis.statistical_h_mean',
        # 'd3m.primitives.tods.feature_analysis.statistical_maximum',
        #'d3m.primitives.tods.feature_analysis.statistical_minimum',
        #'d3m.primitives.tods.feature_analysis.statistical_mean_abs',
        #'d3m.primitives.tods.feature_analysis.statistical_mean_abs_temporal_derivative',
        #'d3m.primitives.tods.feature_analysis.statistical_mean_temporal_derivative',
        #'d3m.primitives.tods.feature_analysis.statistical_median_abs_deviation',
        #'d3m.primitives.tods.feature_analysis.statistical_kurtosis',
        #'d3m.primitives.tods.feature_analysis.statistical_skew',
        #'d3m.primitives.tods.feature_analysis.statistical_std',
        #'d3m.primitives.tods.feature_analysis.statistical_var',
        #'d3m.primitives.tods.feature_analysis.statistical_variation',
        #'d3m.primitives.tods.feature_analysis.statistical_vec_sum',
        #'d3m.primitives.tods.feature_analysis.statistical_willison_amplitude',
        #'d3m.primitives.tods.feature_analysis.statistical_zero_crossing',
        #'d3m.primitives.tods.feature_analysis.spectral_residual_transform',
        #'d3m.primitives.tods.feature_analysis.fast_fourier_transform',
        #'d3m.primitives.tods.feature_analysis.discrete_cosine_transform',
        #'d3m.primitives.tods.feature_analysis.non_negative_matrix_factorization',
        #'d3m.primitives.tods.feature_analysis.bk_filter',
        #'d3m.primitives.tods.feature_analysis.hp_filter',
        #'d3m.primitives.tods.feature_analysis.truncated_svd',
        #'d3m.primitives.tods.feature_analysis.wavelet_transform',
        #'d3m.primitives.tods.feature_analysis.trmf',
    ],
    'detection_algorithm': [
        'd3m.primitives.tods.detection_algorithm.pyod_ae',
        # 'd3m.primitives.tods.detection_algorithm.pyod_vae',
        'd3m.primitives.tods.detection_algorithm.pyod_cof',
        # 'd3m.primitives.tods.detection_algorithm.pyod_sod',
        # 'd3m.primitives.tods.detection_algorithm.pyod_abod',
        # 'd3m.primitives.tods.detection_algorithm.pyod_hbos',
        # 'd3m.primitives.tods.detection_algorithm.pyod_iforest',
        #'d3m.primitives.tods.detection_algorithm.pyod_lof',
        #'d3m.primitives.tods.detection_algorithm.pyod_knn',
        #'d3m.primitives.tods.detection_algorithm.pyod_ocsvm',
        #'d3m.primitives.tods.detection_algorithm.pyod_loda',
        #'d3m.primitives.tods.detection_algorithm.pyod_cblof',
        #'d3m.primitives.tods.detection_algorithm.pyod_sogaal',
        #'d3m.primitives.tods.detection_algorithm.pyod_mogaal',
        #'d3m.primitives.tods.detection_algorithm.matrix_profile',
        #'d3m.primitives.tods.detection_algorithm.AutoRegODetector',
        #'d3m.primitives.tods.detection_algorithm.LSTMODetector',
        #'d3m.primitives.tods.detection_algorithm.AutoRegODetector',
        #'d3m.primitives.tods.detection_algorithm.PCAODetector',
        #'d3m.primitives.tods.detection_algorithm.KDiscordODetector',
        #'d3m.primitives.tods.detection_algorithm.deeplog',
        #'d3m.primitives.tods.detection_algorithm.telemanom',
    ],
    'contamination': [0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2],
}


def _rank_first_metric(pipeline_result): # pragma: no cover
    if pipeline_result.status == 'COMPLETED':
        scores = pipeline_result.scores
        pipeline_result.rank = -scores['value'][0]
        return pipeline_result
    else:
        # error
        pipeline_result.rank = 1
        return pipeline_result

def _generate_data_preparation_params(): # pragma: no cover
    from axolotl.utils import schemas as schemas_utils
    data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']
    return data_preparation_params
    
def _generate_scoring_pipeline(): # pragma: no cover
    from axolotl.utils import schemas as schemas_utils
    scoring_pipeline = schemas_utils.get_scoring_pipeline()
    return scoring_pipeline
    
def _generate_data_preparation_pipeline(): # pragma: no cover
    from axolotl.utils import schemas as schemas_utils
    data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
    return data_preparation_pipeline

def _generate_pipline(combinations): # pragma: no cover
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep

    piplines = []
    for combination in combinations:
        # Creating pipeline
        pipeline_description = Pipeline()
        pipeline_description.add_input(name='inputs')
        
        # The first three steps are fixed
        # Step 0: dataset_to_dataframe
        # step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: column_parser
        # step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

        # Step 2: extract_columns_by_semantic_types(attributes)
        # step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        step_2.add_output('produce')
        step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                                                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
        pipeline_description.add_step(step_2)

        # Step 3: extract_columns_by_semantic_types(targets)
        # step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
        step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
        step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_3.add_output('produce')
        step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                                                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        pipeline_description.add_step(step_3)

        attributes = 'steps.2.produce'
        targets = 'steps.3.produce'
        # print('-------------------------------------------------')
        # print(combination[0])
        # print('-------------------------------------------------')
        # print(len(combination))


        # for j in range(len(combination)):

        # for k in range(len(combination)):
        #     exec(f'cat_{k} = k*2')
        #     print(cat_0)

        # print(list(combination[0][1].keys())[0])
        # print(list(combination[0][1].values())[0])



















        tods_step_4 = PrimitiveStep(primitive=index.get_primitive(combination[0][0]))
        for i in range(len(combination[0][1])):
            if list(combination[1][1].values())[i] != "None":
                tods_step_4.add_hyperparameter(name=str(list(combination[0][1].keys())[i]), argument_type=ArgumentType.VALUE, data=list(combination[0][1].values())[i])
            else:
                tods_step_4.add_hyperparameter(name=str(list(combination[0][1].keys())[i]), argument_type=ArgumentType.VALUE, data=None)
        tods_step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
        tods_step_4.add_output('produce')
        pipeline_description.add_step(tods_step_4)

        tods_step_5 = PrimitiveStep(primitive=index.get_primitive(combination[1][0]))
        for i in range(len(combination[1][1])):
            if list(combination[1][1].values())[i] != "None":
                tods_step_5.add_hyperparameter(name=str(list(combination[1][1].keys())[i]), argument_type=ArgumentType.VALUE, data=list(combination[1][1].values())[i])
            else:
                tods_step_5.add_hyperparameter(name=str(list(combination[1][1].keys())[i]), argument_type=ArgumentType.VALUE, data=None)
        tods_step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
        tods_step_5.add_output('produce')
        pipeline_description.add_step(tods_step_5)

        tods_step_6 = PrimitiveStep(primitive=index.get_primitive(combination[2][0]))
        for i in range(len(combination[2][1])):
            if list(combination[2][1].values())[i] != "None":
                tods_step_6.add_hyperparameter(name=str(list(combination[2][1].keys())[i]), argument_type=ArgumentType.VALUE, data=list(combination[2][1].values())[i])
            else:
                tods_step_6.add_hyperparameter(name=str(list(combination[2][1].keys())[i]), argument_type=ArgumentType.VALUE, data=None)
        tods_step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
        tods_step_6.add_output('produce')
        # tods_step_6.add_hyperparameter(name='contamination', argument_type=ArgumentType.VALUE, data=combination[2][1]['contamination'])
        pipeline_description.add_step(tods_step_6)

        #tods_step_7 = PrimitiveStep(primitive=index.get_primitive(combination[3]))
        #tods_step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
        #tods_step_7.add_output('produce')
        #pipeline_description.add_step(tods_step_7)

        # Finalize the pipeline
        # final_step = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
        final_step = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
        final_step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
        final_step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        final_step.add_output('produce')
        pipeline_description.add_step(final_step)

        pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')
        
        pipeline_description.id = str(uuid.uuid4())
        pipeline_description.created = Pipeline().created
        data = pipeline_description.to_json()
        # print(data)
        # print('--------------------------------------------------------------------------------------------------------------------------')

        piplines.append(pipeline_description)
    return piplines

def get_component_from_json(path):
    import json

    components = []

    f = open(path)
    data = json.load(f)

    for i in data:
        components.append(str(i))

    f.close()
    return components

def get_primitives_from_json(path):
    import json
    import itertools

    components = []

    f = open(path)
    data = json.load(f)

    space = []

    for i in data:
        # print(i)
        p = []
        if str(i) != 'contamination':
            for j in data[i]:
                temp = 'd3m.primitives.tods.' + i  + '.' + j
                # print(temp)
                # p.append(temp)
                hyperparams = []
                hyperparams_name = []
                for k in data[i][j]:
                    hyperparams.append(data[i][j][k])
                    hyperparams_name.append(k)
                # print(hyperparams_name)
                combination = list(itertools.product(*hyperparams))
                # print(combination)
                # print("---------------")
                # print(list(itertools.product(*hyperparams)))
                # print(list(itertools.product(*hyperparams))[0][0])
                for comb in combination:
                    # print(temp, comb)
                    # for m in hyperparams_name:
                    #     print(m, combination[hyperparams_name.index(m)])
                    one = {}
                    for m in range(len(comb)):
                        # print(hyperparams_name[m], comb[m])
                        # one_comp[]
                        # one.append(hyperparams_name[m])
                        # one.append(comb[m])
                        one[hyperparams_name[m]] = comb[m]
                    # print(one)
                    p.append((temp, one))



            space.append(p)
        elif str(i) == 'contamination':
            p.append(data[i])
            space.append(data[i])

    # print(space)
    # for i in space:
    #     print(i)

    return space





def _generate_pipelines(primitive_python_paths, cpu_count=40): # pragma: no cover
    """
    Args:
        primitive_python_paths: a list of primitive Python paths for algorithms
    
    Returns:
        the pipline description json
    """
    import itertools
    import multiprocessing as mp

    #components = ['data_processing', 'timeseries_processing', 'feature_analysis', 'detection_algorithm']
    components = ['timeseries_processing', 'feature_analysis', 'detection_algorithm', 'contamination']
    combinations = itertools.product(*(primitive_python_paths[k] for k in components))


    # print(get_component_from_json('test.json'))

    # print(components)
    # print(combinations)

    # for k in components:
    #     print(primitive_python_paths[k])

    # data = get_primitives_from_json('test.json')

    # for k in get_component_from_json('test.json'):
    #     print(data[k])

    space = get_primitives_from_json('test2.json')

    combinations = itertools.product(*(i for i in space))

    for i in combinations:
        print(i)

    return _generate_pipline(combinations)
    #pipelines = []

    ## Allocate tasks
    #combination_each_core_list = [[] for i in range(cpu_count)]
    #for idx, combination in enumerate(combinations):
    #    core = idx % cpu_count
    #    combination_each_core_list[core].append(combination)

    ## Obtain all the pipelines
    #pool = mp.Pool(processes=cpu_count)
    #results = [pool.apply_async(_generate_pipline,
    #                            args=(combinations,))
    #           for combinations in combination_each_core_list]
    #piplines = []
    #for p in results:
    #    piplines.extend(p.get())

    return piplines

temp = _generate_pipelines(primitive_python_paths, 40)

# print(_generate_pipelines(primitive_python_paths, 40))
