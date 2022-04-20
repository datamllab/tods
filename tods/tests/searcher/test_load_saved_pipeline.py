import os
import d3m
import unittest
import pandas as pd
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from tods import generate_dataset, fit_pipeline, save_fitted_pipeline, load_fitted_pipeline

from tods.detection_algorithm.core.UODCommonTest import UODCommonTest

from pyod.utils.data import generate_data
from d3m.container import DataFrame as d3m_dataframe
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive

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

step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition'))
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

class testLoadSavedPipeline(unittest.TestCase):
	def test_load_saved_happyCase(self):
		self.fitted_pipeline = fit_pipeline(dataset, pipeline_description, 'F1_MACRO')
		fitted_pipeline_id = save_fitted_pipeline(self.fitted_pipeline)

		loaded_pipeline = load_fitted_pipeline(fitted_pipeline_id)
		print(loaded_pipeline['runtime'])
		steps_state = loaded_pipeline['runtime'].steps_state
		print(steps_state)
		autoencoder_model = steps_state[5]['clf_']
		vae_model = steps_state[6]['clf_']
		LSTMODetector = steps_state[7]['clf_']
		deeplog = steps_state[8]['clf_']
		pyod_mogaal = steps_state[9]['clf_']
		pyod_sogaal = steps_state[10]['clf_']
		telemanom = steps_state[11]['clf_']


		assert (hasattr(autoencoder_model, 'decision_scores_') and
				autoencoder_model.decision_scores_ is not None)
		assert (hasattr(autoencoder_model, 'labels_') and
				autoencoder_model.labels_ is not None)
		assert (hasattr(autoencoder_model, 'threshold_') and
				autoencoder_model.threshold_ is not None)
		assert (hasattr(autoencoder_model, '_mu') and
				autoencoder_model._mu is not None)
		assert (hasattr(autoencoder_model, '_sigma') and
				autoencoder_model._sigma is not None)

		assert (hasattr(vae_model, 'decision_scores_') and
				vae_model.decision_scores_ is not None)
		assert (hasattr(vae_model, 'labels_') and
				vae_model.labels_ is not None)
		assert (hasattr(vae_model, 'threshold_') and
				vae_model.threshold_ is not None)
		assert (hasattr(vae_model, '_mu') and
				vae_model._mu is not None)
		assert (hasattr(vae_model, '_sigma') and
				vae_model._sigma is not None)

		assert (hasattr(pyod_mogaal, 'decision_scores_') and
				pyod_mogaal.decision_scores_ is not None)
		assert (hasattr(pyod_mogaal, 'labels_') and
				pyod_mogaal.labels_ is not None)
		assert (hasattr(pyod_mogaal, 'threshold_') and
				pyod_mogaal.threshold_ is not None)
		assert (hasattr(pyod_mogaal, '_mu') and
				pyod_mogaal._mu is not None)
		assert (hasattr(pyod_mogaal, '_sigma') and
				pyod_mogaal._sigma is not None)

		assert (hasattr(pyod_sogaal, 'decision_scores_') and
				pyod_sogaal.decision_scores_ is not None)
		assert (hasattr(pyod_sogaal, 'labels_') and
				pyod_sogaal.labels_ is not None)
		assert (hasattr(pyod_sogaal, 'threshold_') and
				pyod_sogaal.threshold_ is not None)
		assert (hasattr(pyod_sogaal, '_mu') and
				pyod_sogaal._mu is not None)
		assert (hasattr(pyod_sogaal, '_sigma') and
				pyod_sogaal._sigma is not None)

		assert (hasattr(LSTMODetector, 'decision_scores_') and
				LSTMODetector.decision_scores_ is not None)
		assert (hasattr(LSTMODetector, 'labels_') and
				LSTMODetector.labels_ is not None)
		assert (hasattr(LSTMODetector, 'threshold_') and
				LSTMODetector.threshold_ is not None)
		assert (hasattr(LSTMODetector, 'left_inds_') and
				LSTMODetector.left_inds_ is not None)
		assert (hasattr(LSTMODetector, 'right_inds_') and
				LSTMODetector.right_inds_ is not None)
		assert (hasattr(LSTMODetector, '_mu') and
				LSTMODetector._mu is not None)
		assert (hasattr(LSTMODetector, '_sigma') and
				LSTMODetector._sigma is not None)

		assert(str(loaded_pipeline['runtime'].pipeline.steps[0].primitive) == "d3m.primitives.tods.data_processing.dataset_to_dataframe")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[1].primitive) == "d3m.primitives.tods.data_processing.column_parser")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[2].primitive) == "d3m.primitives.tods.data_processing.extract_columns_by_semantic_types")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[3].primitive) == "d3m.primitives.tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition")

		assert(str(loaded_pipeline['runtime'].pipeline.steps[4].primitive) == "d3m.primitives.tods.feature_analysis.fast_fourier_transform")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[5].primitive) == "d3m.primitives.tods.detection_algorithm.pyod_ae")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[6].primitive) == "d3m.primitives.tods.detection_algorithm.pyod_vae")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[7].primitive) == "d3m.primitives.tods.detection_algorithm.LSTMODetector")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[8].primitive) == "d3m.primitives.tods.detection_algorithm.deeplog")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[9].primitive) == "d3m.primitives.tods.detection_algorithm.pyod_mogaal")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[10].primitive) == "d3m.primitives.tods.detection_algorithm.pyod_sogaal")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[11].primitive) == "d3m.primitives.tods.detection_algorithm.telemanom")
		assert(str(loaded_pipeline['runtime'].pipeline.steps[12].primitive) == "d3m.primitives.tods.data_processing.construct_predictions")

if __name__ == '__main__':
	unittest.main()
