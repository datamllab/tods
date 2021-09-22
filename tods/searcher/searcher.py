from ray import tune

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

from tods import generate_dataset, load_pipeline, evaluate_pipeline

class RaySearcher():
  def __init__(self, dataset, metric):
    self.dataset = dataset
    self.metric = metric

  def search(self, search_space, config):
    if config["searching_algorithm"] == "random":
      from ray.tune.suggest.basic_variant import BasicVariantGenerator
      searcher = BasicVariantGenerator() #Random/Grid Searcher
    elif config["searching_algorithm"] == "hyperopt":
      from ray.tune.suggest.hyperopt import HyperOptSearch
      searcher = HyperOptSearch(max_concurrent=2, metric="accuracy") #HyperOpt Searcher
    else:
      raise ValueError("Searching algorithm not supported.")

    analysis = tune.run(
      self._evaluate,
      config = search_space,
      num_samples = config["num_samples"],
      resources_per_trial = {"cpu": 4, "gpu": 0},
      mode = 'max',
      search_alg = searcher,
      name = config["searching_algorithm"] + "_" + str(config["num_samples"])
    )

    best_config = analysis.get_best_config(metric="accuracy")
    self.clearer_best_config(best_config)
    return best_config

  def _evaluate(self, search_space):
    pipeline = self.build_pipeline(search_space)
    pipeline_result = evaluate_pipeline(self.dataset, pipeline, self.metric)
    score = pipeline_result.scores.value[0]
    tune.report(accuracy=score)

  def build_pipeline(self, search_space):
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep
    import sys

    primitive_map = {'axiswise_scaler': 'transformation',
    'standard_scaler': 'transformation',
    'power_transformer': 'transformation',
    'quantile_transformer': 'transformation',
    'moving_average_transform': 'transformation',
    'simple_exponential_smoothing': 'transformation',
    'holt_smoothing': 'transformation',
    'holt_winters_exponential_smoothing': 'transformation',
    'time_series_seasonality_trend_decomposition': 'decomposition',
    'subsequence_segmentation': ''
    }







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



    if 'timeseries_processing' in search_space.keys():
      timeseries_processing_list = []

      timeseries_processing = search_space.pop('timeseries_processing', None)
      if ' ' in timeseries_processing:
        timeseries_processing_list = timeseries_processing.split(' ')
      else:
        timeseries_processing_list.append(timeseries_processing)

      for x in range(len(timeseries_processing_list)):
        this = sys.modules[__name__]
        name = 'step_' + str(counter)
        setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.' + primitive_map[timeseries_processing_list[x]] + '.' +  timeseries_processing_list[x])))
        this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.' + primitive_map[timeseries_processing_list[x]] + '.' +  timeseries_processing_list[x]))

        this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
        for key, value in search_space.items():
          if timeseries_processing_list[x] in key:
            hp_name = key.replace(timeseries_processing_list[x] + '_', '')
            if value == "None":
              this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=None)
            elif value == "True":
              this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=True)
            elif value == "False":
              this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=False)
            else:
              this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
        this.name.add_output('produce')
        pipeline_description.add_step(this.name)
        counter += 1






    feature_analysis_list = []

    feature_analysis = search_space.pop('feature_analysis', None)
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
      for key, value in search_space.items():
        if feature_analysis_list[x] in key:
          hp_name = key.replace(feature_analysis_list[x] + '_', '')
          if value == "None":
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=None)
          elif value == "True":
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=True)
          elif value == "False":
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=False)
          else:
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=value)
      this.name.add_output('produce')
      pipeline_description.add_step(this.name)
      counter += 1





    detection_algorithm_list = []

    detection_algorithm = search_space.pop('detection_algorithm', None)
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
      for key, value in search_space.items():
        if detection_algorithm_list[x] in key:
          hp_name = key.replace(detection_algorithm_list[x] + '_', '')
          if value == "None":
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=None)
          elif value == "True":
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=True)
          elif value == "False":
            this.name.add_hyperparameter(name=hp_name, argument_type=ArgumentType.VALUE, data=False)
          else:
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
    # print(data)
    return pipeline_description

  def clearer_best_config(self, best_config):
    print('the best choice for timeseries_processing is: ', best_config['timeseries_processing'])
    for key, value in best_config.items():
      temp = best_config['timeseries_processing'].split(" ")
      for i in temp:
        if (i + '_') in key:
          print("the best" + key.replace(i + '_', " ") + " for " + 
          i + ": " + str(value))

    print('the best choice for feature analysis is: ', best_config['feature_analysis'])
    for key, value in best_config.items():
      temp = best_config['feature_analysis'].split(" ")
      for i in temp:
        if (i + '_') in key:
          print("the best" + key.replace(i + '_', " ") + " for " + 
          i + ": " + str(value))

    print('the best choice for detection algorithm is: ', best_config['detection_algorithm'])
    for key, value in best_config.items():
      temp = best_config['detection_algorithm'].split(" ")
      for i in temp:
        if (i + '_') in key:
          print("the best" + key.replace(i + '_', " ") + " for " + 
          i + ": " + str(value))



def datapath_to_dataset(path, target_index):
  df = pd.read_csv(path)
  return generate_dataset(df, target_index)

def json_to_searchspace(path, config, use_all_combination, ignore_hyperparams):
  import json

  with open(path) as f:
    data = json.load(f)
  
  def get_all_comb(stuff):
    import itertools
    temp = []
    for L in range(0, len(stuff)+1):
      for subset in itertools.permutations(stuff, L):
        subset = list(subset)
        temp2 = ''
        for i in subset:
          temp2 = temp2 + (i + ' ')
        temp2 = temp2[:-1]
        if temp2 != '':
          temp.append(temp2)
    return temp

  search_space = {}
  from itertools import permutations
  for primitive_type, primitive_list in data.items():
    temp = []
    if not ignore_hyperparams:
      for primitive_name, hyperparams in primitive_list.items():
        temp.append(primitive_name)
        for hyperparams_name, hyperparams_value in hyperparams.items():
          name = primitive_name + '_' + hyperparams_name
          if config['searching_algorithm'] == 'hyperopt':
            search_space[name] = tune.choice(hyperparams_value)
          else:
            search_space[name] = tune.grid_search(hyperparams_value)
    if use_all_combination == True:
      if config['searching_algorithm'] == 'hyperopt':
        search_space[primitive_type] = tune.choice(get_all_comb(temp))
      else:
        search_space[primitive_type] = tune.grid_search(get_all_comb(temp))
    elif use_all_combination == False:
      if config['searching_algorithm'] == 'hyperopt':
        search_space[primitive_type] = tune.choice(temp)
      else:
        search_space[primitive_type] = tune.grid_search(temp)

  return search_space
