import uuid
import random
import pandas as pd
import json
from pprint import pprint
from sklearn.datasets import make_classification

from d3m import container
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import TaskKeyword, PerformanceMetric

from axolotl.utils import data_problem
from axolotl.backend.simple import SimpleRunner
# from axolotl.backend.ray import RayRunner
# from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import pipeline as pipeline_utils, schemas as schemas_utils

import tods
from tods.search import BruteForceSearch

table_path = 'datasets/anomaly/yahoo_sub_5/yahoo_sub_5_dataset/tables/learningData.csv'
df = pd.read_csv(table_path)
dataset, problem_description = data_problem.generate_dataset_problem(df,
                                                                     target_index=7,
                                                                     task_keywords=[TaskKeyword.ANOMALY_DETECTION,],
                                                                     performance_metrics=[{'metric': PerformanceMetric.F1}])

print(dataset)
print(problem_description)

metrics = [{'metric': PerformanceMetric.F1, 'params': {'pos_label': '1'}},
          ]

pipeline_path = 'example_pipeline.json'
pipeline = pipeline_utils.load_pipeline(pipeline_path)
print(pipeline)

data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
scoring_pipeline = schemas_utils.get_scoring_pipeline()
data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']

backend = SimpleRunner(random_seed=0) 
pipeline_result = backend.evaluate_pipeline(problem_description=problem_description,
                                            pipeline=pipeline,
                                            input_data=[dataset],
                                            metrics=metrics,
                                            data_preparation_pipeline=data_preparation_pipeline,
                                            scoring_pipeline=scoring_pipeline,
                                            data_preparation_params=data_preparation_params)
print(pipeline_result)

