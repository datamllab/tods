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

table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'

search_space_path = 'tods/tests/searcher/test_search_space.json'

with open(search_space_path) as f:
    search_space= json.load(f)

df = pd.read_csv(table_path)
dataset = generate_dataset(df[0:50], 6)

class SearcherTest(unittest.TestCase):
    def test_searcher(self):
        config = {
            "metric":'F1_MACRO',
            "num_samples": 1,
            "mode": 'min',
            "use_all_combinations": False,
            "ignore_hyperparameters":False
            }
        
        # initialize the searcher
        searcher = RaySearcher(dataset=df,
                       metric='ALL',
                       beta=1.0)
        # Start searching
        best_pipeline = {}
        for i in range(10):
            random.seed(4)
            search_result = searcher.search(search_space=search_space,config=config)
        
            self.assertIsInstance(search_result,list)
            hyperparameter_search_result = search_result[1]
            print("hyperparameter_search_result: ")
            if i ==0:
                best_pipeline = hyperparameter_search_result['best_config']
            else:
                self.assertEqual(hyperparameter_search_result['best_config'], best_pipeline)
                                                                        
        
        
if __name__ == '__main__':
    unittest.main()