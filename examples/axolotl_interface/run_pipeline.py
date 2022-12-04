import sys
import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline,json_to_config
from tods.utils import build_pipeline

this_path = os.path.dirname(os.path.abspath(__file__))
default_data_path = os.path.join(this_path, '../../datasets/anomaly/raw_data/yahoo_sub_5.csv')

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
parser.add_argument('--table_path', type=str, default=default_data_path,
                    help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=6,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='F1_MACRO',
                    help='Evaluation Metric (F1, F1_MACRO)')
parser.add_argument('--beta',type=float, default=1,
                    help='Evaluation Metric (F1, F1_MACRO)')
parser.add_argument('--pipeline_path', 
                    default=os.path.join(this_path, '/mnt/tods/examples/test.json'),
                    help='Input the path of the pre-built pipeline description')

args = parser.parse_args()

print(args)

table_path = args.table_path 
target_index = args.target_index # what column is the target
pipeline_path = args.pipeline_path
metric = args.metric # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)


# Load the default pipeline
# config = load_pipeline(pipeline_path)

# Build pipeline using config

# FIXME tods.timeseries_processing.subsequence_segmentation
#  [['auto_correlation']] sk [['bk_filter']] [['non_negative_matrix_factorization']] [['statistical_variation']]  [['wavelet_transform']] 
# [['telemanom']] [['PCAODetector']] [['KDiscordODetector']] [['dagmm']]  
# config = {
#             'timeseries_processing':[
#                     ['standard_scaler',{'with_mean':1}]],
#             'detection_algorithm':[
#                     # ['pyod_ae',{'hidden_neurons':[32,16,8,16,32]}]
#                     ['PCAODetector',]
#                     ],
#             'feature_analysis':[
#                     ['statistical_maximum',{'window_size':3}],
#                     ['statistical_minimum',]], #Specify hyperparams as k,v pairs
# }
#,
# TODO add this hyperparam to run all f_a primitives
            # "hyperparams": {
            #     "contamination": {
            #         "type": "VALUE",
            #         "data": 0.1
            #     },
            #     "use_semantic_types": {
            #         "type": "VALUE",
            #         "data": true
            #     },
            #     "use_columns": {
            #         "type": "VALUE",
            #         "data": [
            #             2
            #         ]
            #     }
            # }
            
config = {    
    'feature_analysis':{
        'non_negative_matrix_factorization':{}
        },      
    'detection_algorithm':{
            'pyod_ae':{
               
                }
        }
                           
    }

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
json1 = json_to_config(config)

default_primitive = {
    'data_processing': [],
    'timeseries_processing': [],
    'feature_analysis': [('statistical_maximum',None)],
    'detection_algorithm': [('pyod_ae', None)],
}

pipeline = build_pipeline(json1)
# pipeline = {"id": "1ec59240-4763-4870-9457-c4f24d8c8d0c", "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json", "created": "2022-12-03T09:51:08.951192Z", "inputs": [{"name": "inputs"}], "outputs": [{"data": "steps.4.produce", "name": "output predictions"}], "steps": [{"type": "PRIMITIVE", "primitive": {"id": "c78138d9-9377-31dc-aee8-83d9df049c60", "version": "0.3.0", "python_path": "d3m.primitives.tods.data_processing.dataset_to_dataframe", "name": "Extract a DataFrame from a Dataset"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "inputs.0"}}, "outputs": [{"id": "produce"}]}, {"type": "PRIMITIVE", "primitive": {"id": "81235c29-aeb9-3828-911a-1b25319b6998", "version": "0.3.0", "python_path": "d3m.primitives.tods.data_processing.column_parser", "name": "Parses strings into their types"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.0.produce"}}, "outputs": [{"id": "produce"}]}, {"type": "PRIMITIVE", "primitive": {"id": "a996cd89-ddf0-367f-8e7f-8c013cbc2891", "version": "0.3.0", "python_path": "d3m.primitives.tods.data_processing.extract_columns_by_semantic_types", "name": "Extracts columns by semantic type"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.1.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"semantic_types": {"type": "VALUE", "data": ["https://metadata.datadrivendiscovery.org/types/Attribute"]}}}, {"type": "PRIMITIVE", "primitive": {"id": "5d088dac-17cb-3e4f-893a-4f6541c75246", "version": "0.3.0", "python_path": "d3m.primitives.tods.detection_algorithm.PCAODetector", "name": "PCAODetector"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.2.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"contamination": {"type": "VALUE", "data": 0.1}}}, {"type": "PRIMITIVE", "primitive": {"id": "2530840a-07d4-3874-b7d8-9eb5e4ae2bf3", "version": "0.3.0", "python_path": "d3m.primitives.tods.data_processing.construct_predictions", "name": "Construct pipeline predictions output"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.3.produce"}, "reference": {"type": "CONTAINER", "data": "steps.1.produce"}}, "outputs": [{"id": "produce"}]}], "digest": "8a64ebfe63d4e94ef15f20d23ccb6c7bbe58d4b75e1a48bf9d7720f4d6f80d01"}

# print(pipeline.to_json_structure()) 
# pipeline = load_pipeline(args.pipeline_path)
# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline.to_json_structure()) 
print(pipeline_result.__dict__)
#raise pipeline_result.error[0]

