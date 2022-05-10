import os
import d3m
import unittest
import pandas as pd
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from tods import generate_dataset, fit_pipeline, save_fitted_pipeline, load_fitted_pipeline, produce_fitted_pipeline, evaluate_pipeline, load_pipeline

from tods.detection_algorithm.core.UODCommonTest import UODCommonTest

from pyod.utils.data import generate_data
from d3m.container import DataFrame as d3m_dataframe
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive

from d3m.container.pandas import convert_lists, convert_ndarray
from d3m import container, utils

table_path = '../../../datasets/anomaly/raw_data/yahoo_sub_5.csv'
df = pd.read_csv(table_path)
dataset = generate_dataset(df, 6)

pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
																													data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)

step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.time_series_seasonality_trend_decomposition'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
pipeline_description.add_step(step_3)

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.fast_fourier_transform'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_vae'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.LSTMODetector'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.deeplog'))
step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.7.produce')
step_8.add_output('produce')
pipeline_description.add_step(step_8)

step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_mogaal'))
step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.8.produce')
step_9.add_output('produce')
pipeline_description.add_step(step_9)

step_10 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_sogaal'))
step_10.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.9.produce')
step_10.add_output('produce')
pipeline_description.add_step(step_10)

step_11 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.telemanom'))
step_11.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.10.produce')
step_11.add_output('produce')
pipeline_description.add_step(step_11)

step_12 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_12.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.11.produce')
step_12.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_12.add_output('produce')
pipeline_description.add_step(step_12)

pipeline_description.add_output(name='output predictions', data_reference='steps.12.produce')
descrip = pipeline_description.to_json()

class testProducePipeline(unittest.TestCase):
	def test_Produce_Pipeline(self):
		self.fitted_pipeline = fit_pipeline(dataset, pipeline_description, 'F1_MACRO')
		fitted_pipeline_id = save_fitted_pipeline(self.fitted_pipeline)
		loaded_pipeline = load_fitted_pipeline(fitted_pipeline_id)
		pipeline_result = produce_fitted_pipeline(dataset, loaded_pipeline)

		temp = evaluate_pipeline(dataset, pipeline_description)

		assert(list(pd.DataFrame(pipeline_result.output.select_columns([1]))) == 
		list(pd.DataFrame(temp.outputs[0]['outputs.0'].select_columns([1]))))
if __name__ == '__main__':
	unittest.main()