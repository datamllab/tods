import unittest
from numpy import random
from d3m.metadata import base as metadata_base
from tods.searcher.searcher import RaySearcher
from d3m import container, utils
import argparse
import os
import ray
from tods import generate_problem,build_pipeline,get_primitives_hyperparam_list,json_to_config,get_evaluate_metric,build_step,generate_dataset, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline

import pandas as pd

import json

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata.problem import Problem
from d3m.container.dataset import Dataset
import sys

from tods.searcher import RaySearcher
from tods.utils import get_primitive_list

config = {
            'timeseries_processing':[
                    ['standard_scaler',{'with_mean':True}]],
            'detection_algorithm':[
                    ['pyod_ae',{'hidden_neurons':[32,16,8,16,32]}]],
            'feature_analysis':[
                    ['statistical_maximum',{'window_size':3}],
                    ['statistical_minimum',]], #Specify hyperparams as k,v pairs
        }
# dataset
# table_path = '../../../datasets/anomaly/raw_data/yahoo_sub_5.csv'
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df[0:20], 6)
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

# search space
search_space = {'timeseries_processing': {'time_series_seasonality_trend_decomposition': {'use_semantic_types': [1, 0]}, 'moving_average_transform': {'window_size': [3, 4, 5], 'norm': ['l1', 'l2', 'max'], 'use_semantic_types': [0, 1]}}, 'feature_analysis': {'statistical_h_mean': {'window_size': [10, 5]}, 'statistical_maximum': {'window_size': [10, 5]}, 'statistical_minimum': {'window_size': [10, 5]}}, 'detection_algorithm': {'pyod_ae': {'dropout_rate': [0.1, 0.2]}, 'pyod_loda': {'n_bins': [10, 15]}, 'pyod_cof': {'n_neighborss': [15, 10]}}}




class SearcherTest(unittest.TestCase):
    def test_searcher_simple_searchspace(self):
        config = {
            "metric":'F1_MACRO',
            "num_samples": 1,
            "mode": 'min',
            "use_all_combinations": False,
            "ignore_hyperparameters":False
            }
        
        # initialize the searcher
        searcher = RaySearcher(dataframe=dataframe,
                               target_index=6,
                               dataset=dataset,
                                metric='ALL',
                                beta=1.0)
        # Start searching
        best_pipeline = {}
        for i in range(5):
            random.seed(4)
            search_result = searcher.search(search_space=search_space,config=config)
        
            self.assertIsInstance(search_result,list)
            hyperparameter_search_result = search_result[1]
            print("hyperparameter_search_result: ")
            if i ==0:
                best_pipeline = hyperparameter_search_result['best_config']
            else:
                self.assertEqual(hyperparameter_search_result['best_config'], best_pipeline)
                
    def test_searcher_exhaustive_searchspace(self):
        config = {
            "metric":'F1_MACRO',
            "num_samples": 1,
            "mode": 'min',
            "use_all_combinations": True,
            "ignore_hyperparameters":True
            }
        
        # initialize the searcher
        searcher = RaySearcher(dataframe=dataframe,
                               target_index=6,
                               dataset=dataset,
                                metric='ALL',
                                beta=1.0)
        # Start searching
        best_pipeline = {}
        for i in range(5):
            random.seed(4)
            search_result = searcher.search(search_space=search_space,config=config)
        
            self.assertIsInstance(search_result,dict)
            print("hyperparameter_search_result: ")
            if i ==0:
                best_pipeline = search_result['best_config']
            else:
                self.assertEqual(search_result['best_config'], best_pipeline)                                                      
        
        
if __name__ == '__main__':
    unittest.main()
