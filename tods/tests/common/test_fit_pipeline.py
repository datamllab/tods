import d3m
import unittest
import pandas as pd
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from tods import generate_dataset, fit_pipeline, evaluate_pipeline, load_pipeline
from d3m import container, utils

table_path = 'yahoo_sub_5.csv'
df = pd.read_csv(table_path)
# dataset = container.DataFrame({'a': [1,12183,0.0,3.7166666666667,5,2109,0]
# , 'z': [2,12715,0.091757964510557,3.6108333333333,60,3229,0]
# , 'b': [3,12736,0.17229675238449998,3.4813888888889,88,3637,0]
# , 'c': [4,12716,0.22621935431999,3.3802777777778,84,1982,0]
# , 'd': [5,12739,0.17635798469946,3.1933333333333,111,2751,0]
# , 'e': [6,12737,0.090491245476051,2.7866666666667004,112,2128,0]
# , 'f': [7,12857,0.08460994072769001,2.4627777777777995,1235,2109,0]
# , 'g': [8,12884,0.06842699169496,2.2541666666667,710,2328,0]
# , 'h': [9,12894,0.13330269689422,2.1180555555556,618,2453,0]
# , 'i': [10,12675,0.085026586189321,2.0691666666667,84,2847,0]
# , 'j': [11,13260,0.097073068447328,2.1972222222222,100,3659,0]
# , 'k': [12,13470,0.0,2.3188888888889,125,5207,0]
# },
# columns=['timestamp', 'value_0', 'value_1', 'value_2', 'value_3', 'value_4', 'anomaly'])

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

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

step_5= PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.telemanom'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')
descrip = pipeline_description.to_json()

class testFitPipeline(unittest.TestCase):
  def testFit_happyCase(self):
    self.fitted_pipeline = fit_pipeline(dataset, pipeline_description, 'F1_MACRO')

    self.assertIsInstance(self.fitted_pipeline, dict)
    self.assertIsInstance(self.fitted_pipeline['runtime'], d3m.runtime.Runtime)
    self.assertIsInstance(self.fitted_pipeline['dataset_metadata'], d3m.metadata.base.DataMetadata)

    assert(self.fitted_pipeline is not None)

if __name__ == '__main__':
  unittest.main()