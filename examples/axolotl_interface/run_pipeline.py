import sys
import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline
from tods.d3m_utils import build_pipeline

this_path = os.path.dirname(os.path.abspath(__file__))
default_data_path = os.path.join(this_path, '../../datasets/anomaly/raw_data/yahoo_sub_5.csv')

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
parser.add_argument('--table_path', type=str, default=default_data_path,
                    help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=6,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='ALL',
                    help='Evaluation Metric (F1, F1_MACRO, RECALL, PRECISION, ALL)')
parser.add_argument('--pipeline_path', 
                    default=os.path.join(this_path, './example_pipelines/autoencoder_pipeline.json'),
                    help='Input the path of the pre-built pipeline description')
parser.add_argument('--alg', type=str, default='pyod_ae',
            choices=['dagmm', 'PCAODetector', 'pyod_ae'])
parser.add_argument('--process', type=str, default='statistical_maximum',
            choices=['statistical_maximum'])


args = parser.parse_args()

config = {
    "algorithm": args.alg,
    "processing": args.process,
    "hidden_neurons": [32,16,8,16,32]
    
}

table_path = args.table_path 
target_index = args.target_index # what column is the target
pipeline_path = args.pipeline_path
metric = args.metric # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = build_pipeline(config)
print("Here", pipeline)

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result.scores)
#raise pipeline_result.error[0]

