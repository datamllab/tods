import os
import d3m
import unittest
import pandas as pd
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from tods import generate_dataset, fit_pipeline, save_fitted_pipeline

dataframe = pd.DataFrame(data=[[1,12183,0,3.716666667,5,2109,0],
                    [2,12715,0.091757965,3.610833333,60,3229,0],
                    [3,12736,0.172296752,3.481388889,88,3637,0],
                    [4,12716,0.226219354,3.380277778,84,1982,0],
                    [6,12737,0.090491245,2.786666667,112,2128,0],
                    [7,12857,0.084609941,2.462777778,1235,2109,0],
                    [8,12884,0.068426992,2.254166667,710,2328,0],
                    [9,12894,0.133302697,2.118055556,618,2453,0],
                    [10,12675,0.085026586,2.069166667,84,2847,0],
                    [11,13260,0.097073068,2.197222222,100,3659,0],
                    [12,13470,0,2.318888889,125,5207,0],
                    [13,13060,0.031063768,2.34,114,5146,0],
                    [14,12949,0.017732751,2.490277778,145,4712,0],
                    [15,13035,0.063354504,2.643888889,91,6363,0],
                    [16,12980,0.087870392,2.848611111,94,5010,0],
                    [17,13677,0.115468157,2.883333333,79,3956,0],
                    [18,13381,0.073413458,2.880833333,50,4063,0],
                    [19,12737,0.040392585,2.900555556,39,3748,0],
                    [20,12554,0.089113356,3.085555556,28,3047,0],
                   ],
             columns = ["timestamp","value_0","value_1","value_2","value_3","value_4","anomaly"],
             )
dataset = generate_dataset(dataframe,6)

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

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
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

pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')
descrip = pipeline_description.to_json()

class testSaveFittedPipeline(unittest.TestCase):
  def test_save_fitted_happyCase(self):
    self.fitted_pipeline = fit_pipeline(dataset, pipeline_description, 'F1_MACRO')
    fitted_pipeline_id = save_fitted_pipeline(self.fitted_pipeline)

    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/fitted_pipeline.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/orders.pkl'))


    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/AutoEncoder.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/DeeplogLstm.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/Detector.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/LSTMOutlierDetector.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/MO_GAAL.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/SO_GAAL.pkl'))
    self.assertTrue(os.path.exists('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/VAE.pkl'))

    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/AutoEncoder'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/DeeplogLstm'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/Detector'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/LSTMOutlierDetector'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/MO_GAAL_discriminator'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/SO_GAAL_discriminator'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/SO_GAAL_generator'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/SO_GAAL_combine_model'))
    self.assertTrue(os.path.isdir('../../../fitted_pipelines/' + str(fitted_pipeline_id) + '/model/VAE'))




if __name__ == '__main__':
  unittest.main()