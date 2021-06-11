import sys
import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline

RATIO = 20


this_path = os.path.dirname(os.path.abspath(__file__))
# default_data_path = os.path.join(this_path, '../datasets/anomaly/raw_data/yahoo_sub_5.csv')
# default_data_path = os.path.join(this_path, 'dataset/point_contextual_0.05.csv')

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
# parser.add_argument('--table_path', type=str, default=os.path.join(this_path, 'dataset/point_contextual_0.05.csv'),
#                     help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=5,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='ALL',
                    help='Evaluation Metric (F1, F1_MACRO)')
parser.add_argument('--pipeline_path', 
                    default=os.path.join(this_path, 'mogaal_pipeline_k7_con0.1.json'),
                    help='Input the path of the pre-built pipeline description')

result_container = []
args = parser.parse_args()

df = pd.read_csv(os.path.join(this_path, '../multidataset/01.csv'))
dataset = generate_dataset(df, args.target_index)
pipeline = load_pipeline(args.pipeline_path)


pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
result_container.append(list(pipeline_result.scores['value']))
# print(pipeline_result)
# print(pipeline_result.scores)
# for i in list(pipeline_result.scores['value']):
#     print(i)

df = pd.read_csv(os.path.join(this_path, '../multidataset/12.csv'))
dataset = generate_dataset(df, args.target_index)
pipeline = load_pipeline(args.pipeline_path)

pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
result_container.append(list(pipeline_result.scores['value']))


df = pd.read_csv(os.path.join(this_path, '../multidataset/23.csv'))
dataset = generate_dataset(df, args.target_index)
pipeline = load_pipeline(args.pipeline_path)

pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
result_container.append(list(pipeline_result.scores['value']))




df = pd.read_csv(os.path.join(this_path, '../multidataset/34.csv'))
dataset = generate_dataset(df, args.target_index)
pipeline = load_pipeline(args.pipeline_path)


pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
result_container.append(list(pipeline_result.scores['value']))


for i in result_container:
    for j in i:
        print(j)
    # print('\n')