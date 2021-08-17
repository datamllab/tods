from ray import tune
# A Brute-Force Search
import uuid
import random

from d3m.metadata.pipeline import Pipeline

from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import  schemas as schemas_utils


import pandas as pd

from tods import schemas as schemas_utils
from tods import generate_dataset, evaluate_pipeline

import os

import argparse

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from axolotl.backend.simple import SimpleRunner
from tods import generate_dataset, generate_problem
# from tods.searcher import BruteForceSearch

from tods import generate_dataset, load_pipeline, evaluate_pipeline

def build_pipeline():
  from d3m import index
  from d3m.metadata.base import ArgumentType
  from d3m.metadata.pipeline import Pipeline, PrimitiveStep

  pipeline_description = Pipeline()
  pipeline_description.add_input(name='inputs')

  counter = 0

  # Step 0: dataset_to_dataframe
  step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
  step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
  step_0.add_output('produce')
  pipeline_description.add_step(step_0)
  counter += 1

  # Step 1: column_parser
  step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
  step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
  step_1.add_output('produce')
  pipeline_description.add_step(step_1)
  counter += 1

  # Step 2: extract_columns_by_semantic_types(attributes)
  step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
  step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
  step_2.add_output('produce')
  step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
  pipeline_description.add_step(step_2)
  counter += 1

  attributes = 'steps.2.produce'
  targets = 'steps.3.produce'

  # step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.holt_smoothing'))
  # step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  # step_3.add_hyperparameter(name="exclude_columns", argument_type=ArgumentType.VALUE, data = (2, 3))
  # step_3.add_hyperparameter(name="use_semantic_types", argument_type=ArgumentType.VALUE, data = True)
  # step_3.add_output('produce')
  # pipeline_description.add_step(step_3)
  # counter += 1

  # step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.time_interval_transform'))
  # step_3.add_hyperparameter(name="time_interval", argument_type=ArgumentType.VALUE, data = 'T')
  # step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  # step_3.add_output('produce')
  # pipeline_description.add_step(step_3)
  # counter += 1

  # # Step 4: processing
  # step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
  # step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  # step_4.add_output('produce')
  # pipeline_description.add_step(step_4)

  #Step 5: Video primitive
  # feature_analysis = config.pop('feature_analysis', None)
  # # print(type(algorithm))
  # alg_python_path = 'd3m.primitives.tods.feature_analysis.' + feature_analysis
  # # alg_python_path = 'd3m.primitives.tods.feature_analysis.statistical_h_mean'
  # step_4 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
  # step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  # # step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # # step_5.add_output('produce')
  # # Add hyperparameters
  # for key, value in config.items():
  #   if feature_analysis in key:
  #     hp_name = key.replace(feature_analysis + '_', '')
  #     step_4.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
  # # for key, value in config.items():
  # #     step_4.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
  # step_4.add_output('produce')
  # pipeline_description.add_step(step_4)

  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)

  # detection_algorithm = config.pop('detection_algorithm', None)
  # # print(type(algorithm))
  # alg_python_path = 'd3m.primitives.tods.detection_algorithm.' + detection_algorithm
  # # alg_python_path = 'd3m.primitives.tods.feature_analysis.statistical_h_mean'
  # step_5 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # # step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # # step_5.add_output('produce')
  # # Add hyperparameters
  # # for key, value in config.items():
  # #   if detecti
  # # for key, value in config.items():
  # #     step_4.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)

  temp = ['statistical_h_mean', 'statistical_minimum']

  import sys
  
  for x in range(len(temp)):
    this = sys.modules[__name__]
    name = 'step_' + str(counter)
    print(name)
    setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.' + temp[x])))
    this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.' + temp[x]))

    this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
    this.name.add_output('produce')
    pipeline_description.add_step(this.name)
    counter += 1

  # for i in range(2):
  #   temp = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_h_mean'))
  #   temp.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  #   temp.add_output('produce')
  #   pipeline_description.add_step(temp)

  temp2 = ["pyod_ae", "pyod_loda"]

  for x in range(len(temp2)):
    this = sys.modules[__name__]
    name = 'step_' + str(counter)
    print(name)
    setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.' + temp2[x])))
    this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.' + temp2[x]))

    this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
    this.name.add_output('produce')
    pipeline_description.add_step(this.name)
    counter += 1







  # # Step 5: algorithm
  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)
  # counter += 1

  # for i in range(1):
  #   temp = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_h_mean'))
  #   temp.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  #   temp.add_output('produce')
  #   pipeline_description.add_step(temp)

  for i in range(1):
    this = sys.modules[__name__]
    name = 'step_' + str(counter)
    print(name)
    setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions')))
    this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))

    this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
    this.name.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    this.name.add_output('produce')
    pipeline_description.add_step(this.name)
    counter += 1


  # # Step 6: Predictions
  # step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
  # step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
  # step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
  # step_6.add_output('produce')
  # pipeline_description.add_step(step_6)
  # counter += 1


  pipeline_description.add_output(name='output predictions', data_reference='steps.' + str(counter - 1) + '.produce')
  data = pipeline_description.to_json()
  print(data)
  return pipeline_description


table_path = '../../yahoo_sub_5.csv'
target_index = 6
metric = 'F1_MACRO'
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

pipeline = build_pipeline()

pipeline_result = evaluate_pipeline(dataset, pipeline, metric)

print(pipeline_result)
