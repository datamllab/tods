import os
import d3m
import unittest
import pandas as pd
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from tods import build_pipeline,generate_dataset, fit_pipeline, save_fitted_pipeline, load_fitted_pipeline, produce_fitted_pipeline, evaluate_pipeline, load_pipeline

from tods.detection_algorithm.core.UODCommonTest import UODCommonTest

from pyod.utils.data import generate_data
from d3m.container import DataFrame as d3m_dataframe
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive

from d3m.container.pandas import convert_lists, convert_ndarray
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

config = {
            'timeseries_processing':[
                    ['standard_scaler',{'with_mean':True}]],
            'detection_algorithm':[
                    ['pyod_ae',{'hidden_neurons':[32,16,8,16,32]}]],
            'feature_analysis':[
                    ('statistical_maximum',{'window_size':3}),
                    ('statistical_minimum',)], #Specify hyperparams as k,v pairs
}


default_primitive = {
    'data_processing': [],
    'timeseries_processing': [],
    'feature_analysis': [('statistical_maximum',None)],
    'detection_algorithm': [('pyod_ae', None)],
}

pipeline_description = build_pipeline(config)

class testProducePipeline(unittest.TestCase):
	def test_Produce_Pipeline(self):
    #  TODO return value of fit_pipeline
		self.fitted_pipeline = fit_pipeline(dataset, pipeline_description, 'F1_MACRO')
		fitted_pipeline_id = save_fitted_pipeline(self.fitted_pipeline[0])
		loaded_pipeline = load_fitted_pipeline(fitted_pipeline_id)
		pipeline_result = produce_fitted_pipeline(dataset, loaded_pipeline)

		temp = evaluate_pipeline(dataset, pipeline_description)

		assert(list(pd.DataFrame(pipeline_result.output.select_columns([1]))) == 
		list(pd.DataFrame(temp.outputs[0]['outputs.0'].select_columns([1]))))
if __name__ == '__main__':
	unittest.main()
