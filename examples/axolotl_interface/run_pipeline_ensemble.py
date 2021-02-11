import sys
import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline

this_path = os.path.dirname(os.path.abspath(__file__))
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
#parser.add_argument('--table_path', type=str, default=os.path.join(this_path, '../datasets/yahoo_sub_5.csv'),
#                    help='Input the path of the input data table')

parser.add_argument('--table_path', type=str, default=os.path.join(this_path, '../datasets/anomaly/yahoo_system_sub_5/yahoo_system_sub_5_dataset/tables/learningData.csv'),
                    help='Input the path of the input data table')

parser.add_argument('--target_index', type=int, default=4,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='F1_MACRO',
                    help='Evaluation Metric (F1, F1_MACRO)')

#parser.add_argument('--pipeline_path', default=os.path.join(this_path, '../tods/resources/default_pipeline.json'),
#                    help='Input the path of the pre-built pipeline description')

#Using the pipeline that was build and saved in example_pipeline
parser.add_argument('--pipeline_path', default=os.path.join(this_path, './example_pipeline.json'),
                    help='Input the path of the pre-built pipeline description')


args = parser.parse_args()

table_path = args.table_path 
target_index = args.target_index # what column is the target
pipeline_path = args.pipeline_path
metric = args.metric # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = load_pipeline(pipeline_path)

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)

