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

table_path = 'datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv'
df = pd.read_csv(table_path)
dataset, problem_description = data_problem.generate_dataset_problem(df,
                                                                     target_index=3,
                                                                     task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                     performance_metrics=[{'metric': PerformanceMetric.F1}])

print(dataset)
print(problem_description)


backend = SimpleRunner(random_seed=0) 
search = BruteForceSearch(problem_description=problem_description, backend=backend)
print(search)
