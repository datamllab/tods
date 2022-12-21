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
                    default=os.path.join(this_path, './test.json'),
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

pipeline = build_pipeline(config)

pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result.scores)
#raise pipeline_result.error[0]

