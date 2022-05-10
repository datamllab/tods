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
  [60,13023,0.0,2.55,58,2148,0],
  [61,13022,0.032756244361869,2.7302777777778,63,1978,0],
  [62,13033,0.054893891024455,2.8169444444444003,61,2021,0],
  [63,13024,0.068514114108229,2.9247222222222,55,2060,0],
  [64,13048,0.05279414163165401,2.8911111111111003,71,2096],
  [65,13740,0.023853017353212,2.9575,64,2082,0],
  [66,13540,0.07426125441559799,2.9080555555556,92,2175,0],
  [67,12724,0.024228588329879,3.0088888888889,44,2332,0],
  [68,13070,0.09233413002519697,3.2033333333333,35,2147,0],
  [69,13106,0.15930655332113,3.6213888888889,53,2163,0],
  [70,13025,0.12755838225296,4.0322222222222,49,2406,0],
  [71,13074,0.10152541717054,4.1227777777778,49,2022,0],
  [72,13079,0.040148453968243986,3.9736111111111,103,2188,],
  [73,13184,0.087208372094752,3.8425,107,2758,0],
  [74,13194,0.074209918996797,3.7097222222222,74,2925,0],
  [75,13191,0.059044537369404015,3.6258333333333,56,3223,0],
  [76,13059,0.06248169832921499,3.4705555555556,60,2507,0],
  [77,13169,0.08876527685714597,3.2877777777778,73,2435,0],
  [78,13114,0.051354431854972,2.9286111111111004,99,2552,0],
  [79,13037,0.074790104163639,2.4888888888889,84,2540,0],
  [80,13179,0.091817341555971,2.2744444444444,129,2642,0],
  [81,13152,0.14762794333026005,2.1733333333333,101,2254,0],
  [82,13095,0.07101004447510299,2.3416666666667,101,2539,0],
  [83,13144,0.07689756334240598,2.3808333333333,51,2596,0],
  [84,13170,0.08412575787388403,2.4663888888889,95,2573,0],
  [85,13162,0.06328921386603299,2.6608333333333,48,2302,0],
  [86,13117,0.057393902128707,2.7558333333333,40,2991,0],
  [87,13129,0.041819399065704,2.8636111111111004,55,3141,0],
  [88,13386,0.073729686380986,2.7586111111111005,56,3285,0],
  [89,13929,0.15365285617975,2.7377777777778,935,3807,0],
  [90,13385,0.060355859742407016,2.6961111111111005,34,2892,0],
  [91,13106,0.10644586288975,2.8569444444444,57,2538,0],
  [92,13113,0.059314286360126985,3.1833333333333,70,2234,0],
  [93,13155,0.096293806236591,3.5544444444444,72,2707,0],
  [94,13186,0.085101425467407,3.8894444444444,66,2382,0],
  [95,13151,0.11149072274185,4.1138888888889,72,2426,0],
  [96,13156,0.076266981262989,3.9519444444444,49,2451,0],
  [97,12813,0.097952120177625,3.8275,41,2288,0],
  [98,12821,0.17250021935572,3.6438888888889,42,2256,0],
  [99,12867,0.11389182319254,3.5608333333333,39,2884,0],
  [100,12837,0.08999961787521,3.5013888888889,81,2398,0],
  [101,12911,0.048649372449385005,3.3088888888889,90,2239,0],
  [102,12842,0.13861764684085998,2.9063888888889,92,2248,0],
  [103,12905,0.1088795585287,2.5027777777777995,81,2387,0],
  [104,12993,0.054235162564995,2.2466666666667003,145,3876,0],
  [105,12974,0.0390040506742,2.1869444444444,47,3073,0],
  [106,13039,0.0744713077811,2.2402777777778,63,3113,0],
  [107,13322,0.040258943675435,2.3727777777778,118,3363,0],
  [108,13606,0.0,2.4566666666667003,56,3796,0],
  [109,13536,0.027955712584728,2.5452777777777995,127,4924,0],
  [110,13341,0.047309968420241,2.6830555555556,48,4300,0],
  [111,13360,0.016602764360002,2.805,114,5225,0],
  [112,13450,0.042432577628353986,2.7386111111111004,78,4047,0],
  [113,14102,0.051191743726563,2.7438888888888995,58,4134,0],
  [114,14026,0.0,2.7586111111111005,56,4786,0],
  [115,13162,0.056724832354639,2.9013888888889,67,4184,0],
  [116,13118,0.055771058827737,3.19,155,2888,0],
  [117,12953,0.081014772096658,3.5561111111111003,123,2674,0],
  [118,12854,0.08253629738290899,3.8433333333333,118,2574,0],
  [119,12952,0.11499203730886,4.0319444444444,133,3123,0],
  [120,12915,0.07668513845109799,3.8844444444444,75,3369,0],
  [121,11994,0.070057457403873,3.6908333333333,29,3284,0],
  [122,11868,0.07031477357556501,3.6141666666667,68,2127,0],
  [123,11977,0.091946448716499,3.5019444444444,91,2117,0],
  [124,11874,0.14560588482235998,3.4205555555556,101,2271,0],
  [125,11913,0.094774329323472,3.1780555555556,22,2513,0],
  [126,11933,0.10217989327054,2.8361111111111,20,2746,0],
  [127,11844,0.04854243074027901,2.5222222222222004,27,2076,0],
  [128,11968,0.068760549683423,2.2416666666667004,45,2297,0],
  [129,11996,0.075440683881139,2.1588888888889,42,2312,0],
  [130,12006,0.11771339431815,2.2763888888889,59,2834,0],
  [131,12225,0.069437397660265,2.3391666666667,52,3584,0],
  [132,12482,0.0,2.4841666666667,62,4009,0]
]

df = pd.DataFrame (data, columns = ['timestamp','value_0','value_1','value_2','value_3','value_4','anomaly'])
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
		assert(str(loaded_pipeline['runtime'].pipeline.steps[3].primitive) == "d3m.primitives.tods.timeseries_processing.time_series_seasonality_trend_decomposition")

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