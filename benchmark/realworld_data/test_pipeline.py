import sys
import argparse
import os
import pandas as pd
from warnings import filterwarnings 
filterwarnings("ignore") 

from tods import generate_dataset, load_pipeline, evaluate_pipeline

this_path = os.path.dirname(os.path.abspath(__file__))
#default_data_path = os.path.join(this_path, './data/web_attack.csv')
#default_data_path = os.path.join(this_path, './data/swan_sf.csv')
default_data_path = os.path.join(this_path, './data/shuttle.csv')

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
parser.add_argument('--data_path', type=str, default=default_data_path,
                    help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=0,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric', type=str, default='ALL',
        help='Evaluation Metric (F1, F1_MACRO, RECALL, PRECISION, ALL)')
parser.add_argument('--pipeline_path', 
                    default=os.path.join(this_path, './pipeline2/autoencoder_pipeline.json'),
                    help='Input the path of the pre-built pipeline description')

args = parser.parse_args()

table_path = args.data_path 
target_index = args.target_index # what column is the target
metric = args.metric # F1 on both label 0 and 1
pipeline_path = args.pipeline_path

output_name = pipeline_path.split("/")[-1][:-5]
output_folder = table_path.split("/")[-1].split(".")[0]

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = load_pipeline(pipeline_path)

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
attrs = pipeline_result.__dict__
print(pipeline_result)
#print(attrs)
raise pipeline_result.error[0]

