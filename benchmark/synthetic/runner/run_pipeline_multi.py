import sys
import argparse
import os
import pandas as pd

from tods import generate_dataset, load_pipeline, evaluate_pipeline


this_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Arguments for running predefined pipelin.')
parser.add_argument('--target_index', type=int, default=5,
                    help='Index of the ground truth (for evaluation)')
parser.add_argument('--metric',type=str, default='ALL',
                    help='Evaluation Metric (F1, F1_MACRO)')
parser.add_argument('--pipeline_path_name', 
                    default='ocsvm_subseg',
                    help='Input the path of the pre-built pipeline description')

args = parser.parse_args()

result_container = []
pipeline_path_list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25]
dataset_list = ['0', '1', '2', '3', '4', '01', '12', '23', '34', '012', '123', '234', '0123', '1234', '01234']


for dat, pip in zip(dataset_list, pipeline_path_list):
    df = pd.read_csv(os.path.join(this_path, '../multidataset/' + dat + '.csv'))
    dataset = generate_dataset(df, args.target_index)
    pipeline_path = os.path.join(this_path, '..', 'Pipeline', args.pipeline_path_name, args.pipeline_path_name + '_con' + str(pip) + '.json')
    pipeline = load_pipeline(pipeline_path)
    pipeline_result = evaluate_pipeline(dataset, pipeline, args.metric)

    result_path = os.path.join('./result', 'multidataset', dat, args.pipeline_path_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    pipeline_result.scores.to_csv(result_path + '/' + args.pipeline_path_name + str(pip) + ".csv")
    result_container.append(list(pipeline_result.scores['value']))


for i in result_container:
    for j in i:
        print(j)
    # print('\n')