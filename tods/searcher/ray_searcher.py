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

def build_pipeline(config):
  from d3m import index
  from d3m.metadata.base import ArgumentType
  from d3m.metadata.pipeline import Pipeline, PrimitiveStep

  pipeline_description = Pipeline()
  pipeline_description.add_input(name='inputs')

  # Step 0: dataset_to_dataframe
  step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
  step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
  step_0.add_output('produce')
  pipeline_description.add_step(step_0)

  # Step 1: column_parser
  step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
  step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
  step_1.add_output('produce')
  pipeline_description.add_step(step_1)

  # Step 2: extract_columns_by_semantic_types(attributes)
  step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
  step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
  step_2.add_output('produce')
  step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
  pipeline_description.add_step(step_2)

  # Step 3: extract_columns_by_semantic_types(targets)
  step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
  step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
  step_3.add_output('produce')
  step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
  pipeline_description.add_step(step_3)

  attributes = 'steps.2.produce'
  targets = 'steps.3.produce'


  # Step 4: processing
  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)

  #Step 5: Video primitive
  feature_analysis = config.pop('feature_analysis', None)
  # print(type(algorithm))
  alg_python_path = 'd3m.primitives.tods.feature_analysis.' + feature_analysis
  # alg_python_path = 'd3m.primitives.tods.feature_analysis.statistical_h_mean'
  step_4 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
  step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
  # step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # Add hyperparameters
  for key, value in config.items():
    if feature_analysis in key:
      hp_name = key.replace(feature_analysis + '_', '')
      step_4.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
  # for key, value in config.items():
  #     step_4.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
  step_4.add_output('produce')
  pipeline_description.add_step(step_4)

  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)

  detection_algorithm = config.pop('detection_algorithm', None)
  # print(type(algorithm))
  alg_python_path = 'd3m.primitives.tods.detection_algorithm.' + detection_algorithm
  # alg_python_path = 'd3m.primitives.tods.feature_analysis.statistical_h_mean'
  step_5 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
  step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # Add hyperparameters
  # for key, value in config.items():
  #   if detecti
  # for key, value in config.items():
  #     step_4.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
  step_5.add_output('produce')
  pipeline_description.add_step(step_5)

  # Step 5: algorithm
  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)

  # Step 6: Predictions
  step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
  step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
  step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
  step_6.add_output('produce')
  pipeline_description.add_step(step_6)


  pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')
  data = pipeline_description.to_json()
  # print(data)
  return pipeline_description

def build_pipeline2(config):
  from d3m import index
  from d3m.metadata.base import ArgumentType
  from d3m.metadata.pipeline import Pipeline, PrimitiveStep

  pipeline_description = Pipeline()
  pipeline_description.add_input(name='inputs')

  step_list = []

  # Step 0: dataset_to_dataframe
  step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
  step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
  step_0.add_output('produce')
  pipeline_description.add_step(step_0)
  step_list.append(step_0)

  # Step 1: column_parser
  step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
  step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
  step_1.add_output('produce')
  pipeline_description.add_step(step_1)
  step_list.append(step_1)

  # Step 2: extract_columns_by_semantic_types(attributes)
  step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
  step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
  step_2.add_output('produce')
  step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
  pipeline_description.add_step(step_2)
  step_list.append(step_2)

  # Step 3: extract_columns_by_semantic_types(targets)
  step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
  step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
  step_3.add_output('produce')
  step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
  pipeline_description.add_step(step_3)
  step_list.append(step_3)

  attributes = 'steps.2.produce'
  targets = 'steps.3.produce'


  # Step 4: processing
  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)
  count = 3

  #Step 5: Video primitive
  feature_analysis = config.pop('feature_analysis', None)
  # print(type(algorithm))
  print('---------------------------------------------------------------------------------------------------------------------------------------------------')
  print(feature_analysis)
  lst = feature_analysis.split()
  for i in lst:
    name = 'd3m.primitives.tods.feature_analysis.' + i
    # setattr(this, 'step_%s' % (count + 1), PrimitiveStep(primitive=index.get_primitive(name)))
    temp = PrimitiveStep(primitive=index.get_primitive(name))
    temp.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    temp.add_output('produce')
    pipeline_description.add_step(temp)
    step_list.append(temp)


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
  cur = len(step_list) - 1

  detection_algorithm = config.pop('detection_algorithm', None)
  # print(type(algorithm))
  alg_python_path = 'd3m.primitives.tods.detection_algorithm.' + detection_algorithm
  # alg_python_path = 'd3m.primitives.tods.feature_analysis.statistical_h_mean'
  step_5 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
  t = 'step.' + str(cur) +  '.produce'
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=t)
  step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # Add hyperparameters
  # for key, value in config.items():
  #   if detecti
  # for key, value in config.items():
  #     step_4.add_hyperparameter(name=key, argument_type=ArgumentType.VALUE, data=value)
  step_5.add_output('produce')
  pipeline_description.add_step(step_5)

  # Step 5: algorithm
  # step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
  # step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
  # step_5.add_output('produce')
  # pipeline_description.add_step(step_5)

  # Step 6: Predictions
  step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
  step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
  step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
  step_6.add_output('produce')
  pipeline_description.add_step(step_6)


  pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')
  data = pipeline_description.to_json()
  # print(data)
  return pipeline_description

def build_pipeline3(config):
  from d3m import index
  from d3m.metadata.base import ArgumentType
  from d3m.metadata.pipeline import Pipeline, PrimitiveStep
  import sys

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




  # data_processing_list = []

  # data_processing = config.pop('data_processing', None)
  # if ' ' in data_processing:
  #   data_processing_list = data_processing.split(' ')
  # else:
  #   data_processing_list.append(data_processing)

  # for x in range(len(data_processing_list)):
  #   this = sys.modules[__name__]
  #   name = 'step_' + str(counter)
  #   print(name)
  #   setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.' + data_processing_list[x])))
  #   this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.' + data_processing_list[x]))

  #   this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
  #   for key, value in config.items():
  #     if data_processing_list[x] in key:
  #       hp_name = key.replace(data_processing_list[x] + '_', '')
  #       this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
  #   this.name.add_output('produce')
  #   pipeline_description.add_step(this.name)
  #   counter += 1




  timeseries_processing_list = []

  timeseries_processing = config.pop('timeseries_processing', None)
  if ' ' in timeseries_processing:
    timeseries_processing_list = timeseries_processing.split(' ')
  else:
    timeseries_processing_list.append(timeseries_processing)

  for x in range(len(timeseries_processing_list)):
    this = sys.modules[__name__]
    name = 'step_' + str(counter)
    print(name)
    setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.' + timeseries_processing_list[x])))
    this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.' + timeseries_processing_list[x]))

    this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
    for key, value in config.items():
      if timeseries_processing_list[x] in key:
        hp_name = key.replace(timeseries_processing_list[x] + '_', '')
        this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
    this.name.add_output('produce')
    pipeline_description.add_step(this.name)
    counter += 1
  





  feature_analysis_list = []

  feature_analysis = config.pop('feature_analysis', None)
  if ' ' in feature_analysis:
    feature_analysis_list = feature_analysis.split(' ')
  else:
    feature_analysis_list.append(feature_analysis)


  for x in range(len(feature_analysis_list)):
    this = sys.modules[__name__]
    name = 'step_' + str(counter)
    print(name)
    setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.' + feature_analysis_list[x])))
    this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.' + feature_analysis_list[x]))

    this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
    for key, value in config.items():
      if feature_analysis_list[x] in key:
        hp_name = key.replace(feature_analysis_list[x] + '_', '')
        this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
    this.name.add_output('produce')
    pipeline_description.add_step(this.name)
    counter += 1






  detection_algorithm_list = []

  detection_algorithm = config.pop('detection_algorithm', None)
  if ' ' in detection_algorithm:
    detection_algorithm_list = detection_algorithm.split(' ')
  else:
    detection_algorithm_list.append(detection_algorithm)

  for x in range(len(detection_algorithm_list)):
    this = sys.modules[__name__]
    name = 'step_' + str(counter) 
    print(name)
    setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.' + detection_algorithm_list[x])))
    this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.' + detection_algorithm_list[x]))

    this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
    for key, value in config.items():
      if detection_algorithm_list[x] in key:
        hp_name = key.replace(detection_algorithm_list[x] + '_', '')
        this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
    this.name.add_output('produce')
    pipeline_description.add_step(this.name)
    counter += 1







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




  pipeline_description.add_output(name='output predictions', data_reference='steps.' + str(counter - 1) + '.produce')
  data = pipeline_description.to_json()
  print(data)
  return pipeline_description



table_path = '../../yahoo_sub_5.csv'
target_index = 6
metric = 'F1_MACRO'
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

def evaluate(config):
  pipeline = build_pipeline3(config)
  # print(pipeline)

  # table_path = '../../yahoo_sub_5.csv'
  # target_index = 6
  # metric = 'F1_MACRO'

  # df = pd.read_csv(table_path)

  # dataset = generate_dataset(df, target_index)

  
  pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
  # print('printing result----------------------------------------------------------------------------------------')
  # print(pipeline_result)
  score = pipeline_result.scores.value[0]

  tune.report(accuracy=score)



def argsparser():
  parser = argparse.ArgumentParser("Automatically searching hyperparameters for video recognition")
  parser.add_argument('--alg', type=str, default='hyperopt',
          choices=['random', 'hyperopt'])
  parser.add_argument('--num_samples', type=int, default=15)
  parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='0')
  parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/hmdb6/')

  return parser

parser = argsparser()
args = parser.parse_args()

config = {
  "searching_algorithm": args.alg,
  "num_samples": args.num_samples,
}

if config["searching_algorithm"] == "random":
  from ray.tune.suggest.basic_variant import BasicVariantGenerator
  searcher = BasicVariantGenerator() #Random/Grid Searcher
elif config["searching_algorithm"] == "hyperopt":
  from ray.tune.suggest.hyperopt import HyperOptSearch
  searcher = HyperOptSearch(max_concurrent=2, metric="accuracy") #HyperOpt Searcher
else:
  raise ValueError("Searching algorithm not supported.")

search_space = {
  "feature_analysis": tune.choice(["statistical_maximum", 'statistical_h_mean', "hp_filter"]),
  "statistical_h_mean_window_size": tune.choice([10, 20, 30]),
  "hp_filter_lamb": tune.choice([1600, 1800]),
  "detection_algorithm": tune.choice(["pyod_ae", "pyod_loda"])
}

search_space2 = {
  # "data_processing": tune.choice(["time_interval_transform"]),
  # "time_interval_transform_time_interval": tune.choice(["5T", "3T"]),
  "timeseries_processing": tune.choice(["moving_average_transform"]),
  "feature_analysis": tune.choice(["statistical_maximum statistical_h_mean statistical_minimum", "statistical_maximum"]),
  "statistical_h_mean_window_size": tune.choice([10, 20, 30]),
  "detection_algorithm": tune.choice(["pyod_ae pyod_loda", "pyod_loda"]),
  "pyod_loda_n_bins": tune.choice([10, 20, 30])
}

analysis = tune.run(
  evaluate,
  config=search_space2,
  num_samples=config["num_samples"],
  resources_per_trial={"cpu": 2, "gpu": 1},
  mode='max',
  search_alg=searcher,
  name=config["searching_algorithm"]+"_"+str(config["num_samples"])
)
best_config = analysis.get_best_config(metric="accuracy")
print(best_config)


# pipeline = build_pipeline(config)


# table_path = '../../yahoo_sub_5.csv'
# target_index = 6
# metric = 'F1_MACRO'

# df = pd.read_csv(table_path)
# print(df)
