import uuid
import random
import pandas as pd
from pprint import pprint
from sklearn.datasets import make_classification

from d3m import container
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import TaskKeyword, PerformanceMetric

from axolotl.utils import data_problem
from axolotl.backend.simple import SimpleRunner
from axolotl.backend.ray import RayRunner
from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import pipeline as pipeline_utils, schemas as schemas_utils

import tods
from tods.search import BruteForceSearch

table_path = 'datasets/anomaly/yahoo_sub_5/yahoo_sub_5_dataset/tables/learningData.csv'
df = pd.read_csv(table_path)
dataset, problem_description = data_problem.generate_dataset_problem(df,
                                                                     target_index=7,
                                                                     task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                     performance_metrics=[{'metric': PerformanceMetric.F1}])

backend = SimpleRunner(random_seed=0) 
search = BruteForceSearch(problem_description=problem_description, backend=backend)

# Find the best pipeline
best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=10)
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

