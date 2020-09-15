import pandas as pd
import sys
import argparse

from searcher import schemas as schemas_utils
from searcher.utils import generate_dataset_problem, evaluate_pipeline
from axolotl.utils import pipeline as pipeline_utils
import os

this_path = os.path.dirname(os.path.abspath(__file__))
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
parser.add_argument('--table_path', type=str, default=os.path.join(this_path, '../datasets/yahoo_sub_5.csv'),
                    help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=6,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='F1_MACRO',
                    help='Evaluation Metric (F1, F1_MACRO)')
parser.add_argument('--pipeline_path', default=os.path.join(this_path, '../tods/searcher/resources/default_pipeline.json'),
                    help='Input the path of the pre-built pipeline description')

args = parser.parse_args()

table_path = args.table_path 
target_index = args.target_index # what column is the target
pipeline_path = args.pipeline_path
metric = args.metric # F1 on both label 0 and 1

time_limit = 30 # How many seconds you wanna search

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset, problem_description = generate_dataset_problem(df, target_index=target_index, metric=metric)

# Load the default pipeline
pipeline = pipeline_utils.load_pipeline(pipeline_path)

# Run the pipeline
pipeline_result = evaluate_pipeline(problem_description, dataset, pipeline)
print(pipeline_result)

