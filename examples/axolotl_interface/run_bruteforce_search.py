import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods import generate_dataset, generate_problem
from tods.searcher import BruteForceSearch

# Some information
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_GOOG.csv' # The path of the dataset
#target_index = 2 # what column is the target

table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
target_index = 6 # what column is the target
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset
time_limit = 30 # How many seconds you wanna search

#metric = 'F1' # F1 on label 1
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index=target_index)
problem_description = generate_problem(dataset, metric)

# Start backend
backend = SimpleRunner(random_seed=0)

# Start search algorithm
search = BruteForceSearch(problem_description=problem_description,
                          backend=backend)

# Find the best pipeline
best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=time_limit)
best_pipeline = best_runtime.pipeline
best_output = best_pipeline_result.output

# Evaluate the best pipeline
best_scores = search.evaluate(best_pipeline).scores


print('*' * 52)
print('Search History:')
for pipeline_result in search.history:
    print('-' * 52)
    print('Pipeline id:', pipeline_result.pipeline.id)
    print(pipeline_result.scores)
print('*' * 52)

print('')

print('*' * 52)
print('Best pipeline:')
print('-' * 52)
print('Pipeline id:', best_pipeline.id)
print('Pipeline json:', best_pipeline.to_json())
print('Output:')
print(best_output)
print('Scores:')
print(best_scores)
print('*' * 52)


