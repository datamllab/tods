import unittest
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
df = pd.read_csv(table_path)
dataset = generate_dataset(df, 6)

class PipelineTest(unittest.TestCase):
    def test_json_to_config(self):
        
        json_config = {
            'timeseries_processing':{
                'standard_scaler':{
                    'with_mean':
                        True
                    }
            },
            'detection_algorithm':{
                'pyod_ae':{
                    'hidden_neurons':
                        [32,16,8,16,32]
                    }
            },  
            'feature_analysis':{
                'statistical_maximum':{
                    'window_size':3
                    },
                'statistical_minimum':{
                    } 
            },     
        }
        self.json_config = json_to_config(json_config)
        
        self.assertEqual(self.json_config,config)
    
    def test_evaluate_matric(self):
        y_true = [0,1,0,0,1,0,1,0,0]
        y_pred = [0,0,0,0,1,0,1,0,1]
        beta = 0.5
        metric = "ALL"
        
        metrics = get_evaluate_metric(y_true,y_pred,beta,metric)
        
        self.assertIsInstance(metrics,dict)
        self.assertIsInstance(metrics['F_beta'],float)
        self.assertIsInstance(metrics['RECALL'],float)
        self.assertIsInstance(metrics['PRECISION'],float)
        self.assertIsInstance(metrics['F1_MACRO'],float)
        self.assertIsInstance(metrics['F1'],float)
        
    def test_build_step(self):
        arguments={'inputs': 'steps.6.produce'}
        primitive_path='detection_algorithm.pyod_ae'
        hyperparams={'hidden_neurons': [32, 16, 8, 16, 32]}

        primitive_step = {'type': 'PRIMITIVE',
                          'primitive': {'id': '67e7fcdf-d645-3417-9aa4-85cd369487d9',
                                        'version': '0.3.0',
                                        'python_path': 'd3m.primitives.tods.detection_algorithm.pyod_ae',
                                        'name': 'TODS.anomaly_detection_primitives.AutoEncoder'},
                          'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.6.produce'}},
                          'outputs': [{'id': 'produce'}],
                          'hyperparams': {'hidden_neurons': {'type': 'VALUE', 'data': [32, 16, 8, 16, 32]}}}
        step = build_step(primitive_path,arguments, hyperparams)
        
        self.assertIsInstance(step,PrimitiveStep)
        self.assertEqual(step.to_json_structure(),primitive_step)
    
    def test_get_primitive_lists(self):
        config = {
            'timeseries_processing':[
                    ['standard_scaler',{'with_mean':True}]],
            'detection_algorithm':[
                    ['pyod_ae',{'hidden_neurons':[32,16,8,16,32]}]],
            'feature_analysis':[
                    ['statistical_maximum',{'window_size':3}],
                    ['statistical_minimum',]], #Specify hyperparams as k,v pairs
        }
        primitive_list = get_primitive_list(config,"feature_analysis")
        hyperparam_list = get_primitives_hyperparam_list(config,"feature_analysis")
        self.assertIsInstance(primitive_list,list)
        self.assertIsInstance(primitive_list[0],str)
        
        self.assertIsInstance(hyperparam_list,list)
        self.assertIsInstance(hyperparam_list[0],dict)
        

    def test_generate_problem(self):
        self.generated_problem = generate_problem(dataset,'ALL')
        problem = {'performance_metrics': [{'metric': 'PRECISION', 'params': {'pos_label': '1'}}, {'metric': 'RECALL', 'params': {'pos_label': '1'}}, {'metric': 'F1_MACRO', 'params': {}}, {'metric': 'F1', 'params': {'pos_label': '1'}}], 'task_keywords': ['ANOMALY_DETECTION']}

        self.assertIsInstance(self.generated_problem,Problem) 
        self.assertEqual(problem,self.generated_problem['problem'])
        

        
    def test_build_pipeline(self):
        
        pipeline_description = {'id': 'eb985bd7-39c9-431d-9ae8-58608b50c084', 
                                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json',
                                'created': '2022-11-30T08:49:11.299630Z',
                                'inputs': [{'name': 'inputs'}],
                                'outputs': [{'data': 'steps.8.produce', 'name': 'output predictions'}],
                                'steps': [{'type': 'PRIMITIVE', 
                                           'primitive': {'id': 'c78138d9-9377-31dc-aee8-83d9df049c60', 
                                                         'version': '0.3.0', 
                                                         'python_path': 'd3m.primitives.tods.data_processing.dataset_to_dataframe',
                                                         'name': 'Extract a DataFrame from a Dataset'}, 
                                           'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'inputs.0'}},
                                           'outputs': [{'id': 'produce'}]}, 
                                          {'type': 'PRIMITIVE', 
                                           'primitive': {'id': '81235c29-aeb9-3828-911a-1b25319b6998', 
                                                         'version': '0.3.0',
                                                         'python_path': 'd3m.primitives.tods.data_processing.column_parser', 
                                                         'name': 'Parses strings into their types'},
                                           'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.0.produce'}},
                                           'outputs': [{'id': 'produce'}]},
                                          {'type': 'PRIMITIVE', 
                                           'primitive': {'id': 'a996cd89-ddf0-367f-8e7f-8c013cbc2891', 
                                                         'version': '0.3.0',
                                                         'python_path': 'd3m.primitives.tods.data_processing.extract_columns_by_semantic_types',
                                                         'name': 'Extracts columns by semantic type'},
                                           'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.1.produce'}}, 
                                           'outputs': [{'id': 'produce'}], 
                                           'hyperparams': {'semantic_types': {'type': 'VALUE', 'data': ['https://metadata.datadrivendiscovery.org/types/Attribute']}}}, {'type': 'PRIMITIVE', 'primitive': {'id': 'a996cd89-ddf0-367f-8e7f-8c013cbc2891', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.data_processing.extract_columns_by_semantic_types', 'name': 'Extracts columns by semantic type'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.0.produce'}}, 'outputs': [{'id': 'produce'}], 'hyperparams': {'semantic_types': {'type': 'VALUE', 'data': ['https://metadata.datadrivendiscovery.org/types/TrueTarget']}}}, {'type': 'PRIMITIVE', 'primitive': {'id': '3b54b820-d4c4-3a1f-813c-14c2d591e284', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.timeseries_processing.standard_scaler', 'name': 'Standard_scaler'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.2.produce'}}, 'outputs': [{'id': 'produce'}], 'hyperparams': {'with_mean': {'type': 'VALUE', 'data': True}}}, {'type': 'PRIMITIVE', 'primitive': {'id': 'f07ce875-bbc7-36c5-9cc1-ba4bfb7cf48e', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.feature_analysis.statistical_maximum', 'name': 'Time Series Decompostional'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.4.produce'}}, 'outputs': [{'id': 'produce'}], 'hyperparams': {'window_size': {'type': 'VALUE', 'data': 3}}}, {'type': 'PRIMITIVE', 'primitive': {'id': '8fc80903-3eed-3222-8f6d-8df781279fbc', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.feature_analysis.statistical_minimum', 'name': 'Time Series Decompostional'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.5.produce'}}, 'outputs': [{'id': 'produce'}]}, {'type': 'PRIMITIVE', 'primitive': {'id': '67e7fcdf-d645-3417-9aa4-85cd369487d9', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.detection_algorithm.pyod_ae', 'name': 'TODS.anomaly_detection_primitives.AutoEncoder'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.6.produce'}}, 'outputs': [{'id': 'produce'}], 'hyperparams': {'hidden_neurons': {'type': 'VALUE', 'data': [32, 16, 8, 16, 32]}}}, {'type': 'PRIMITIVE', 'primitive': {'id': '2530840a-07d4-3874-b7d8-9eb5e4ae2bf3', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.data_processing.construct_predictions', 'name': 'Construct pipeline predictions output'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.7.produce'}, 'reference': {'type': 'CONTAINER', 'data': 'steps.1.produce'}}, 'outputs': [{'id': 'produce'}]}], 'digest': '98c67cd42352057eb61de362730f5e2d0e3a7d0738f46d5c747ec96e8b2169fe'}
		
        self.built_pipeline = build_pipeline(config)
        
        self.assertIsInstance(self.built_pipeline,Pipeline)
        self.assertEqual(self.built_pipeline.to_json_structure()['steps'],pipeline_description['steps'])
        self.assertEqual(self.built_pipeline.to_json_structure()['schema'],pipeline_description['schema'])
        self.assertEqual(self.built_pipeline.to_json_structure()['inputs'],pipeline_description['inputs'])
        self.assertEqual(self.built_pipeline.to_json_structure()['outputs'],pipeline_description['outputs'])

    def test_generate_problem(self):
        self.generated_dataset = generate_dataset(df,6)
        self.assertIsInstance(self.generated_dataset,Dataset)


        
if __name__ == '__main__':
   unittest.main()