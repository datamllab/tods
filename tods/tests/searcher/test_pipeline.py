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

config_system = {'detection_algorithm': [
    ('pyod_ocsvm',)
],

    'feature_analysis': [
        ('statistical_maximum',),
    ]
}
# table_path = '../../../datasets/anomaly/raw_data/yahoo_sub_5.csv'
# df = pd.read_csv(table_path)
# dataset = generate_dataset(df, 6)
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
        
        pipeline_description = {'id': 'b2e3050c-4447-4305-bacc-968c07f2c719', 
                                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json', 
                                'created': '2022-12-04T14:20:45.549783Z', 
                                'inputs': [{'name': 'inputs'}], 
                                'outputs': [{'data': 'steps.7.produce', 
                                             'name': 'output predictions'}], 
                                'steps': [{'type': 'PRIMITIVE', 
                                           'primitive': {'id': 'c78138d9-9377-31dc-aee8-83d9df049c60', 
                                                         'version': '0.3.0', 
                                                         'python_path': 'd3m.primitives.tods.data_processing.dataset_to_dataframe', 
                                                         'name': 'Extract a DataFrame from a Dataset'}, 
                                           'arguments': {'inputs': {'type': 'CONTAINER', 
                                                                    'data': 'inputs.0'}}, 
                                           'outputs': [{'id': 'produce'}]}, 
                                          {'type': 'PRIMITIVE', 
                                           'primitive': {'id': '81235c29-aeb9-3828-911a-1b25319b6998', 
                                                         'version': '0.3.0', 
                                                         'python_path': 'd3m.primitives.tods.data_processing.column_parser', 
                                                         'name': 'Parses strings into their types'}, 
                                           'arguments': {'inputs': {'type': 'CONTAINER', 
                                                                    'data': 'steps.0.produce'}}, 
                                           'outputs': [{'id': 'produce'}]}, 
                                          {'type': 'PRIMITIVE', 
                                           'primitive': {'id': 'a996cd89-ddf0-367f-8e7f-8c013cbc2891', 
                                                         'version': '0.3.0', 
                                                         'python_path': 'd3m.primitives.tods.data_processing.extract_columns_by_semantic_types', 
                                                         'name': 'Extracts columns by semantic type'}, 
                                           'arguments': {'inputs': 
                                               {'type': 'CONTAINER', 
                                                'data': 'steps.1.produce'}}, 
                                           'outputs': [{'id': 'produce'}], 
                                           'hyperparams': {'semantic_types': 
                                               {'type': 'VALUE', 
                                                'data': ['https://metadata.datadrivendiscovery.org/types/Attribute']}}}, 
                                          {'type': 'PRIMITIVE', 
                                           'primitive': {'id': '3b54b820-d4c4-3a1f-813c-14c2d591e284', 
                                                         'version': '0.3.0', 
                                                         'python_path': 'd3m.primitives.tods.timeseries_processing.standard_scaler', 
                                                         'name': 'Standard_scaler'}, 
                                           'arguments': {'inputs': 
                                               {'type': 'CONTAINER', 
                                                'data': 'steps.2.produce'}}, 
                                           'outputs': [{'id': 'produce'}], 
                                           'hyperparams': {'with_mean': 
                                               {'type': 'VALUE', 
                                                'data': True}}}, {'type': 'PRIMITIVE', 
                                                                  'primitive': {'id': 'f07ce875-bbc7-36c5-9cc1-ba4bfb7cf48e', 
                                                                                'version': '0.3.0', 
                                                                                'python_path': 'd3m.primitives.tods.feature_analysis.statistical_maximum', 
                                                                                'name': 'Time Series Decompostional'}, 
                                                                  'arguments': {'inputs': 
                                                                      {'type': 'CONTAINER', 
                                                                       'data': 'steps.3.produce'}}, 
                                                                  'outputs': [{'id': 'produce'}], 
                                                                  'hyperparams': {'window_size': {'type': 'VALUE', 
                                                                                                  'data': 3}}}, 
                                                {'type': 'PRIMITIVE', 
                                                 'primitive': {'id': '8fc80903-3eed-3222-8f6d-8df781279fbc', 
                                                               'version': '0.3.0', 
                                                               'python_path': 'd3m.primitives.tods.feature_analysis.statistical_minimum', 
                                                               'name': 'Time Series Decompostional'}, 
                                                 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.4.produce'}}, 'outputs': [{'id': 'produce'}]}, {'type': 'PRIMITIVE', 'primitive': {'id': '67e7fcdf-d645-3417-9aa4-85cd369487d9', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.detection_algorithm.pyod_ae', 'name': 'TODS.anomaly_detection_primitives.AutoEncoder'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.5.produce'}}, 'outputs': [{'id': 'produce'}], 'hyperparams': {'hidden_neurons': {'type': 'VALUE', 'data': [32, 16, 8, 16, 32]}}}, {'type': 'PRIMITIVE', 'primitive': {'id': '2530840a-07d4-3874-b7d8-9eb5e4ae2bf3', 'version': '0.3.0', 'python_path': 'd3m.primitives.tods.data_processing.construct_predictions', 'name': 'Construct pipeline predictions output'}, 'arguments': {'inputs': {'type': 'CONTAINER', 'data': 'steps.6.produce'}, 'reference': {'type': 'CONTAINER', 'data': 'steps.1.produce'}}, 'outputs': [{'id': 'produce'}]}], 'digest': '1b1e46038a5ab2f2115f028a6257c68819ad94e7417b65ef125ae241258a757a'}
        self.built_pipeline = build_pipeline(config)
        
        self.assertIsInstance(self.built_pipeline,Pipeline)
        self.assertEqual(self.built_pipeline.to_json_structure()['steps'],pipeline_description['steps'])
        self.assertEqual(self.built_pipeline.to_json_structure()['schema'],pipeline_description['schema'])
        self.assertEqual(self.built_pipeline.to_json_structure()['inputs'],pipeline_description['inputs'])
        self.assertEqual(self.built_pipeline.to_json_structure()['outputs'],pipeline_description['outputs'])

    def build_system_pipeline(self):
        pipeline_description = {
    "id": "73e15443-4ee7-40d5-8b76-a01b06333d50",
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
    "created": "2023-01-30T16:39:07.005212Z",
    "inputs": [
        {
            "name": "inputs"
        }
    ],
    "outputs": [
        {
            "data": "steps.9.produce",
            "name": "output predictions"
        }
    ],
    "steps": [
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "f31f8c1f-d1c5-43e5-a4b2-2ae4a761ef2e",
                "version": "0.2.0",
                "python_path": "d3m.primitives.tods.common.denormalize",
                "name": "Denormalize datasets"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "inputs.0"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ]
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "c78138d9-9377-31dc-aee8-83d9df049c60",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.data_processing.dataset_to_dataframe",
                "name": "Extract a DataFrame from a Dataset"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.0.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ]
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "989562ac-b50f-4462-99cb-abef80d765b2",
                "version": "0.1.0",
                "python_path": "d3m.primitives.tods.common.csv_reader",
                "name": "Columns CSV reader"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.1.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ],
            "hyperparams": {
                "use_columns": {
                    "type": "VALUE",
                    "data": [
                        0,
                        1
                    ]
                },
                "return_result": {
                    "type": "VALUE",
                    "data": "replace"
                }
            }
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "81235c29-aeb9-3828-911a-1b25319b6998",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.data_processing.column_parser",
                "name": "Parses strings into their types"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.2.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ],
            "hyperparams": {
                "parse_semantic_types": {
                    "type": "VALUE",
                    "data": [
                        "http://schema.org/Boolean",
                        "http://schema.org/Integer",
                        "http://schema.org/Float",
                        "https://metadata.datadrivendiscovery.org/types/FloatVector"
                    ]
                }
            }
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "a996cd89-ddf0-367f-8e7f-8c013cbc2891",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.data_processing.extract_columns_by_semantic_types",
                "name": "Extracts columns by semantic type"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.3.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ],
            "hyperparams": {
                "semantic_types": {
                    "type": "VALUE",
                    "data": [
                        "https://metadata.datadrivendiscovery.org/types/Attribute"
                    ]
                }
            }
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "a996cd89-ddf0-367f-8e7f-8c013cbc2891",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.data_processing.extract_columns_by_semantic_types",
                "name": "Extracts columns by semantic type"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.3.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ],
            "hyperparams": {
                "semantic_types": {
                    "type": "VALUE",
                    "data": [
                        "https://metadata.datadrivendiscovery.org/types/TrueTarget"
                    ]
                }
            }
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "f07ce875-bbc7-36c5-9cc1-ba4bfb7cf48e",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.feature_analysis.statistical_maximum",
                "name": "Time Series Decompostional"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.4.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ]
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "b454adf7-5820-3e6f-8383-619f13fb1cb6",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.detection_algorithm.pyod_ocsvm",
                "name": "TODS.anomaly_detection_primitives.OCSVMPrimitive"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.6.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ]
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "01d36760-235c-3cdd-95dd-3c682c634c49",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.detection_algorithm.system_wise_detection",
                "name": "Sytem_Wise_Anomaly_Detection_Primitive"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.7.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ]
        },
        {
            "type": "PRIMITIVE",
            "primitive": {
                "id": "2530840a-07d4-3874-b7d8-9eb5e4ae2bf3",
                "version": "0.3.0",
                "python_path": "d3m.primitives.tods.data_processing.construct_predictions",
                "name": "Construct pipeline predictions output"
            },
            "arguments": {
                "inputs": {
                    "type": "CONTAINER",
                    "data": "steps.8.produce"
                },
                "reference": {
                    "type": "CONTAINER",
                    "data": "steps.1.produce"
                }
            },
            "outputs": [
                {
                    "id": "produce"
                }
            ]
        }
    ],
    "digest": "193c74a8386c80f5ce81ab8d979eef97f46901cf63c70d45c3b4a2064b3df4c9"
}
        self.built_system_pipeline = build_pipeline(config_system)
        
        self.assertIsInstance(self.built_system_pipeline,Pipeline)
        self.assertEqual(self.built_system_pipeline.to_json_structure()['steps'],pipeline_description['steps'])
        self.assertEqual(self.built_system_pipeline.to_json_structure()['schema'],pipeline_description['schema'])
        self.assertEqual(self.built_system_pipeline.to_json_structure()['inputs'],pipeline_description['inputs'])
        self.assertEqual(self.built_system_pipeline.to_json_structure()['outputs'],pipeline_description['outputs'])
    def test_generate_problem(self):
        self.generated_dataset = generate_dataset(dataframe,6)
        self.assertIsInstance(self.generated_dataset,Dataset)


        
if __name__ == '__main__':
   unittest.main()