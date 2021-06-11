import uuid
import random

from d3m.metadata.pipeline import Pipeline

from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import  schemas as schemas_utils

primitive_python_paths = { # pragma: no cover
    'data_processing': [
        #'d3m.primitives.tods.data_processing.time_interval_transform',
        #'d3m.primitives.tods.data_processing.categorical_to_binary',
        #'d3m.primitives.tods.data_processing.column_filter',
        #'d3m.primitives.tods.data_processing.timestamp_validation',
        #'d3m.primitives.tods.data_processing.duplication_validation',
        #'d3m.primitives.tods.data_processing.continuity_validation',
    ],
    'timeseries_processing': [
        #'d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler',
        #'d3m.primitives.tods.timeseries_processing.subsequence_segmentation',
        'd3m.primitives.tods.timeseries_processing.transformation.standard_scaler',
        #'d3m.primitives.tods.timeseries_processing.transformation.power_transformer',
        #'d3m.primitives.tods.timeseries_processing.transformation.quantile_transformer',
        #'d3m.primitives.tods.timeseries_processing.transformation.moving_average_transform',
        #'d3m.primitives.tods.timeseries_processing.transformation.simple_exponential_smoothing',
        #'d3m.primitives.tods.timeseries_processing.transformation.holt_smoothing',
        #'d3m.primitives.tods.timeseries_processing.transformation.holt_winters_exponential_smoothing',
        #'d3m.primitives.tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition',
    ],
    'feature_analysis': [
        #'d3m.primitives.tods.feature_analysis.auto_correlation',
        #'d3m.primitives.tods.feature_analysis.statistical_mean',
        #'d3m.primitives.tods.feature_analysis.statistical_median',
        #'d3m.primitives.tods.feature_analysis.statistical_g_mean',
        #'d3m.primitives.tods.feature_analysis.statistical_abs_energy',
        #'d3m.primitives.tods.feature_analysis.statistical_abs_sum',
        #'d3m.primitives.tods.feature_analysis.statistical_h_mean',
        #'d3m.primitives.tods.feature_analysis.statistical_maximum',
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
        #'d3m.primitives.tods.detection_algorithm.pyod_ae',
        #'d3m.primitives.tods.detection_algorithm.pyod_vae',
        #'d3m.primitives.tods.detection_algorithm.pyod_cof',
        #'d3m.primitives.tods.detection_algorithm.pyod_sod',
        #'d3m.primitives.tods.detection_algorithm.pyod_abod',
        #'d3m.primitives.tods.detection_algorithm.pyod_hbos',
        'd3m.primitives.tods.detection_algorithm.pyod_iforest',
        #'d3m.primitives.tods.detection_algorithm.pyod_lof',
        #'d3m.primitives.tods.detection_algorithm.pyod_knn',
        'd3m.primitives.tods.detection_algorithm.pyod_ocsvm',
        #'d3m.primitives.tods.detection_algorithm.pyod_loda',
        #'d3m.primitives.tods.detection_algorithm.pyod_cblof',
        'd3m.primitives.tods.detection_algorithm.pyod_sogaal',
        'd3m.primitives.tods.detection_algorithm.pyod_mogaal',
        'd3m.primitives.tods.detection_algorithm.matrix_profile',
        'd3m.primitives.tods.detection_algorithm.AutoRegODetector',
        #'d3m.primitives.tods.detection_algorithm.LSTMODetector',
        #'d3m.primitives.tods.detection_algorithm.AutoRegODetector',
        #'d3m.primitives.tods.detection_algorithm.PCAODetector',
        #'d3m.primitives.tods.detection_algorithm.KDiscordODetector',
        #'d3m.primitives.tods.detection_algorithm.deeplog',
        #'d3m.primitives.tods.detection_algorithm.telemanom',
    ],
    'contamination': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
}

def _generate_pipeline(combinations):
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
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline_description.add_step(step_0)

        # Step 1: column_parser
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
        step_1.add_output('produce')
        pipeline_description.add_step(step_1)

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

        tods_step_4 = PrimitiveStep(primitive=index.get_primitive(combination[0]))
        tods_step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
        #tods_step_4.add_hyperparameter(name='window_size', argument_type=ArgumentType.VALUE, data=10)
        #tods_step_4.add_hyperparameter(name='step', argument_type=ArgumentType.VALUE, data=1)
        tods_step_4.add_output('produce')
        pipeline_description.add_step(tods_step_4)

        tods_step_5= PrimitiveStep(primitive=index.get_primitive(combination[1]))
        tods_step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
        tods_step_5.add_output('produce')
        tods_step_5.add_hyperparameter(name='contamination', argument_type=ArgumentType.VALUE, data=combination[2])
        pipeline_description.add_step(tods_step_5)

        # Finalize the pipeline
        final_step = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
        final_step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
        final_step.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
        final_step.add_output('produce')
        pipeline_description.add_step(final_step)

        pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')
        data = pipeline_description.to_json()
        #with open('../pipelines/'+str(combination[1].split(".")[-1])+'_'+str(combination[2])+".json", 'w') as f:
        with open('./pipelines/simple/'+str(combination[1].split(".")[-1])+'_'+str(combination[2])+".json", 'w') as f:
            f.write(data)
        pipeline_description.id = str(uuid.uuid4())
        pipeline_description.created = Pipeline().created
        piplines.append(pipeline_description)
    return piplines

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
    components = ['timeseries_processing', 'detection_algorithm', 'contamination']
    combinations = itertools.product(*(primitive_python_paths[k] for k in components))


    return _generate_pipeline(combinations)


if __name__ == "__main__":
    combinations = _generate_pipelines(primitive_python_paths)
    print(combinations)
