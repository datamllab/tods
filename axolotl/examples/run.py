import os
import time
from pprint import pprint
import pandas as pd
from sklearn.datasets import make_regression

from d3m import container
from d3m.metadata.pipeline import Pipeline

from axolotl.utils import data_problem, pipeline as pipeline_utils
from axolotl.backend.simple import SimpleRunner
from axolotl.backend.ray import RayRunner
from axolotl.algorithms.random_search import RandomSearch

# init runner
#backend = RayRunner(random_seed=42, volumes_dir=None, n_workers=3)
backend = SimpleRunner(random_seed=42, volumes_dir=None)
#time.sleep(30)

table_path = os.path.join('..', 'tests', 'data', 'datasets', 'iris_dataset_1', 'tables', 'learningData.csv')
df = pd.read_csv(table_path)
dataset, problem_description = data_problem.generate_dataset_problem(df, task='binary_classification', target_index=5) 

# The method fit search for the best pipeline based on the time butget and fit the best pipeline based on the rank with the input_data.
search = RandomSearch(problem_description=problem_description, backend=backend)

fitted_pipeline, fitted_pipelineine_result = search.search_fit(input_data=[dataset], time_limit=30)

produce_results = search.produce(fitted_pipeline, [dataset])

print(produce_results.output)
