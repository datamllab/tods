import sys
import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline



this_path = os.path.dirname(os.path.abspath(__file__))
# default_data_path = os.path.join(this_path, '../datasets/anomaly/raw_data/yahoo_sub_5.csv')
# default_data_path = os.path.join(this_path, 'dataset/point_contextual_0.05.csv')

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
# parser.add_argument('--table_path', type=str, default=os.path.join(this_path, 'dataset/point_contextual_0.05.csv'),
#                     help='Input the path of the input data table')
parser.add_argument('--target_index', type=int, default=1,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='ALL',
                    help='Evaluation Metric (F1, F1_MACRO)')
# parser.add_argument('--pipeline_path', 
#                     default=os.path.join(this_path, 'mp_pipeline.json'),
#                     help='Input the path of the pre-built pipeline description')

args = parser.parse_args()

pipeline_path_name = '../' + 'ocsvm_subseg_con'
pipeline_path_list = [0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25]
dataset_list = [0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2]


result_container = []
for dat, pip in zip(dataset_list, pipeline_path_list):
    pipeline_path = os.path.join(this_path, pipeline_path_name + str(pip) + '.json')

    df = pd.read_csv(os.path.join(this_path, '../unidataset/point_global_' + str(dat) + '.csv'))
    dataset = generate_dataset(df, args.target_index)
    pipeline = load_pipeline(pipeline_path)
    pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
    result_container.append(list(pipeline_result.scores['value']))


for dat, pip in zip(dataset_list, pipeline_path_list):
    pipeline_path = os.path.join(this_path, pipeline_path_name + str(pip) + '.json')

    df = pd.read_csv(os.path.join(this_path, '../unidataset/point_contextual_' + str(dat) + '.csv'))
    dataset = generate_dataset(df, args.target_index)
    pipeline = load_pipeline(pipeline_path)
    pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
    result_container.append(list(pipeline_result.scores['value']))


for dat, pip in zip(dataset_list, pipeline_path_list):
    pipeline_path = os.path.join(this_path, pipeline_path_name + str(pip) + '.json')

    df = pd.read_csv(os.path.join(this_path, '../unidataset/collective_global_' + str(dat) + '.csv'))
    dataset = generate_dataset(df, args.target_index)
    pipeline = load_pipeline(pipeline_path)
    pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
    result_container.append(list(pipeline_result.scores['value']))


for dat, pip in zip(dataset_list, pipeline_path_list):
    pipeline_path = os.path.join(this_path, pipeline_path_name + str(pip) + '.json')

    df = pd.read_csv(os.path.join(this_path, '../unidataset/collective_seasonal_' + str(dat) + '.csv'))
    dataset = generate_dataset(df, args.target_index)
    pipeline = load_pipeline(pipeline_path)
    pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
    result_container.append(list(pipeline_result.scores['value']))


for dat, pip in zip(dataset_list, pipeline_path_list):
    pipeline_path = os.path.join(this_path, pipeline_path_name + str(pip) + '.json')

    df = pd.read_csv(os.path.join(this_path, '../unidataset/collective_trend_' + str(dat) + '.csv'))
    dataset = generate_dataset(df, args.target_index)
    pipeline = load_pipeline(pipeline_path)
    pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)
    result_container.append(list(pipeline_result.scores['value']))


for i in result_container:
    for j in i:
        print(j)
    print('\n')