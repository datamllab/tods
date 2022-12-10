import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods import generate_dataset, generate_problem,save_fitted_pipeline,load_fitted_pipeline,produce_fitted_pipeline, fit_pipeline
from tods.searcher import BruteForceSearch,RaySearcher

import json
# Some information
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_GOOG.csv' # The path of the dataset
#target_index = 2 # what column is the target

table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
search_space_path = "tods/searcher/example_search_space.json"
target_index = 6 # what column is the target
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset
time_limit = 30 # How many seconds you wanna search

#metric = 'F1' # F1 on label 1
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)

dataset = generate_dataset(df, target_index=target_index)
# train_problem_description = generate_problem(dataset, metric)

# initialize the searcher
searcher = RaySearcher(dataframe=df,
                       target_index=6,
                       dataset=dataset,
                       metric='ALL',
                       beta=1.0)

# get JSON search space
with open(search_space_path) as f:
        search_space= json.load(f)

#define search process
config = {
"metric":'F1_MACRO',
"num_samples": 1,
"mode": 'min',
"use_all_combinations": False,
"ignore_hyperparameters":False
}

# Start searching
search_result = searcher.search(search_space=search_space,config=config)

# Output result
if isinstance(search_result,list):
        primitive_search_result = search_result[0]
        hyperparameter_search_result = search_result[1]
        print('*' * 52)
        print('Primitive Search History:')
        print(primitive_search_result['search_history'])
        print('-' * 52)
        print('Hyperparameter Search History:')
        print(hyperparameter_search_result['search_history'])
        print('*' * 52)

        print('*' * 52)
        print('Best pipeline:')
        print('Pipeline id:', hyperparameter_search_result['best_pipeline_id'])
        print('Pipeline json:', hyperparameter_search_result['best_config'])
        print('*' * 52)
        best_pipeline_id = hyperparameter_search_result['best_pipeline_id']
else:
        print('*' * 52)
        print('Primitive Search History:')
        print(search_result['search_history'])
        print('*' * 52)

        print('*' * 52)
        print('Best pipeline:')
        print('Pipeline id:', search_result['best_pipeline_id'])
        print('Pipeline json:', search_result['best_config'])
        print('*' * 52)
        best_pipeline_id = search_result['best_pipeline_id']

# load the fitted pipeline based on id
loaded = load_fitted_pipeline('705f2699-e03b-42bc-a865-d31a11354dae')

print(loaded)
# use the loaded fitted pipelines
result = produce_fitted_pipeline(dataset, loaded)

print('*' * 52)
print("Load model result: ")
print('-' * 52)       
print(result)
print('*' * 52)
