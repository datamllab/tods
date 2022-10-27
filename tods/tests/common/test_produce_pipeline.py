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

data = [
  [1,12183,0.0,3.7166666666667,5,2109,0],
  [2,12715,0.091757964510557,3.6108333333333,60,3229,0],
  [3,12736,0.17229675238449998,3.4813888888889,88,3637,0],
  [4,12716,0.22621935431999,3.3802777777778,84,1982,0],
  [5,12739,0.17635798469946,3.1933333333333,111,2751,0],
  [6,12737,0.090491245476051,2.7866666666667004,112,2128,0],
  [7,12857,0.08460994072769001,2.4627777777777995,1235,2109,0],
  [8,12884,0.06842699169496,2.2541666666667,710,2328,0],
  [9,12894,0.13330269689422,2.1180555555556,618,2453,0],
  [10,12675,0.085026586189321,2.0691666666667,84,2847,0],
  [11,13260,0.097073068447328,2.1972222222222,100,3659,0],
  [12,13470,0.0,2.3188888888889,125,5207,0],
  [13,13060,0.031063767542922,2.34,114,5146,0],
  [14,12949,0.017732750501525,2.4902777777778,145,4712,0],
  [15,13035,0.063354504072079,2.6438888888889,91,6363,0],
  [16,12980,0.087870391896335,2.8486111111111003,94,5010,0],
  [17,13677,0.11546815687729,2.8833333333333,79,3956,0],
  [18,13381,0.073413457727404,2.8808333333333,50,4063,0],
  [19,12737,0.040392584616896,2.9005555555556,39,3748,0],
  [20,12554,0.08911335594722301,3.0855555555556,28,3047,0],
  [21,12470,0.098030053711531,3.3536111111111,29,4099,0],
  [22,12490,0.047140641497552,3.7438888888889,24,2122,0],
  [23,12539,0.10481279080241,3.7947222222222,19,3387,0],
  [24,12530,0.20478886838928,3.801111111111101,21,1950,0],
  [25,13002,0.04485100631921201,3.6508333333333,27,2927,0],
  [26,12989,0.1053622140254,3.555,46,1889,0],
  [27,13038,0.08436887679639,3.4769444444444,133,1910,0],
  [28,13011,0.097980673762982,3.2158333333333,143,3747,0],
  [29,12984,0.10165726215275,3.1141666666667,86,4994,0],
  [30,13079,0.056764513454874,2.7983333333333,118,2009,0],
  [31,13048,0.074428708878932,2.4252777777778,56,2899,0],
  [32,13096,0.091244453451818,2.14,92,2298,0],
  [33,13003,0.094529332881679,1.9822222222222,85,1894,0],
  [34,13057,0.016638011234698,1.9694444444444,122,1999,0],
  [35,13023,0.038096861957006005,2.0741666666667,74,3007,0],
  [36,13033,0.064497814457643,2.2505555555556,84,2838,0],
  [37,13034,0.030426401876334,2.2819444444444,54,4113,0],
  [38,13068,0.095423209955973,2.4216666666667,77,2150,0],
  [39,13057,0.069688744272108,2.5997222222222005,84,3007,0],
  [40,13047,0.03468622413034,2.7544444444444003,139,2484,0],
  [41,13795,0.089564461084836,2.7258333333333,65,2101,0],
  [42,13528,0.07337616196456799,2.8302777777778,38,2001,0],
  [43,13032,0.061939295606039,2.9422222222222,35,2102,0],
  [44,13084,0.11419089175512,3.0919444444444,47,2129,0],
  [45,13000,0.10475925920163,3.3519444444444,37,4422,0],
  [46,13008,0.079657960399444,3.6952777777778,53,4573,0],
  [47,12978,0.14475546275416,3.8269444444444,55,1989,0],
  [48,13067,0.1421711341096,3.7877777777778,45,1953,0],
  [49,13086,0.07696963969656899,3.7536111111111,46,1872,0],
  [50,13023,0.06393273436444799,3.61,35,1850,0],
  [51,13046,0.14973281021845006,3.5091666666667,68,2879,0],
  [52,13032,0.041478839355346,3.4205555555556,82,1840,0],
  [53,13012,0.089317973365284,3.2647222222222,154,2134,0],
  [54,13051,0.088820248166203,2.7944444444444,128,2234,0],
  [55,12979,0.054872994406929,2.46,79,3769,0],
  [56,13025,0.07913553329046401,2.2075,66,2717,0],
  [57,13007,0.16317996709063,2.1758333333333,92,2171,0],
  [58,13036,0.08671926699280201,2.3058333333333,67,2224,0],
  [59,13043,0.0733999511789,2.3983333333333,58,1967,0],
]

df = pd.DataFrame (data, columns = ['timestamp','value_0','value_1','value_2','value_3','value_4','anomaly'])

dataset = generate_dataset(df, 6)
print(dataset)

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

step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')
descrip = pipeline_description.to_json()

class testProducePipeline(unittest.TestCase):
  def test_Produce_Pipeline(self):

    fitted_pipeline = fit_pipeline(dataset, pipeline_description, 'F1_MACRO')
    fitted_pipeline_id = save_fitted_pipeline(fitted_pipeline)
    loaded_pipeline = load_fitted_pipeline(fitted_pipeline_id)
    pipeline_result = produce_fitted_pipeline(dataset, loaded_pipeline)

    temp = evaluate_pipeline(dataset, pipeline_description)

    assert(list(pd.DataFrame(pipeline_result.output).iloc[: , -1]) == 
    list(pd.DataFrame(temp.outputs[0]['outputs.0']).iloc[: , -1]))

if __name__ == '__main__':
  unittest.main()