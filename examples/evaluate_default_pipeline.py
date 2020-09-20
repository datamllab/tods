import pandas as pd

from searcher import schemas as schemas_utils
from searcher.utils import generate_dataset_problem, evaluate_pipeline

table_path = 'datasets/yahoo_sub_5.csv'
target_index = 6 # what column is the target

#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset
time_limit = 30 # How many seconds you wanna search
#metric = 'F1' # F1 on label 1
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset, problem_description = generate_dataset_problem(df, target_index=target_index, metric=metric)

# Load the default pipeline
pipeline = schemas_utils.load_default_pipeline()

# Run the pipeline
pipeline_result = evaluate_pipeline(problem_description, dataset, pipeline)
print(pipeline_result)

