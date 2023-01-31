from ray import tune
import ray
import uuid
import random
from d3m.metadata.pipeline import Pipeline
from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import  schemas as schemas_utils
import pandas as pd
import numpy as np
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
from tods import generate_dataset, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline,get_evaluate_metric,get_primitive_list,get_primitives_hyperparam_list
import pdb
import json
from sklearn.metrics import precision_recall_curve,fbeta_score
from tods.data_processing import DatasetToDataframe
from tods.utils import build_pipeline

@ray.remote
class GlobalStats:

  def __init__(self):
    self.fitted_pipeline_list = []
    self.pipeline_description_list = []
    self.scores = []

  def append_fitted_pipeline_id(self, val):
    self.fitted_pipeline_list.append(val)

  def append_pipeline_description(self, description):
    self.pipeline_description_list.append(description.to_json())

  def append_score(self, score):
    self.scores.append(score)

  def get_fitted_pipeline_list(self):
    return self.fitted_pipeline_list
  
  def get_pipeline_description_list(self):
    return self.pipeline_description_list

  def get_scores(self):
    return self.scores



class RaySearcher():
  """
  A class to tune scalable hypermeters by Ray Tune
  """
  def __init__(self,dataframe,target_index,dataset, metric, beta):
    ray.init(local_mode=True, ignore_reinit_error=True)
    self.dataframe = dataframe
    self.metric = metric
    self.beta = beta
    self.stats = GlobalStats.remote()
    self.dataset = dataset
    # self.dataset = generate_dataset(self.dataframe,target_index)


  def search(self,search_space, config):
    """
    Launch a search process to find a suitable pipeline

    Parameters
    ----------
    search_space:
        A search space transform from a json file user designed
        defines valid values for your hyperparameters and can specify how these values are sampled.
    config:
        An object containing parameters defined by the user to set how the searcher will run.
        Config currently accepts 5 arguments:
        
        - ``searching_algorithm`` : Name of the desired search algorithm to use. https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose
        - ``num_samples`` : Number of times to sample from the hyperparameter space. Defaults to 1. 
        - ``mode`` : Must be one of [min, max]. Determines whether objective is minimizing or maximizing the metric attribute. 
        - ``use_all_combinations`` : Boolean. If True, use exhaustive search all the primitive combination with default hyperparams. If False, use simple search space.
        - ``ignore_hyperparameters`` : Boolean.  If False, search hyperparam combinations of the output of the primitive search process.

    .. code-block:: python

        #define search process
        config = {
        "searching_algorithm": 'nevergrad',
        "num_samples": 1,
        "mode": 'min',
        "use_all_combinations": True,
        "ignore_hyperparameters": False
        }
    
    Returns
    -------
      best_config
      best_config_pipeline_id
    """

    primitive_search_space = self.json_to_primitive_searchspace(search_space, is_exhaustive=config["use_all_combinations"])

    # set the number of CPU cores
    import multiprocessing
    num_cores =  multiprocessing.cpu_count()

    # Launch searcher
    primitive_analysis = ray.tune.run(
      self._evaluate,
      metric = config['metric'],
      config = primitive_search_space,
      num_samples = 1,
      # resources_per_trial = {"cpu": num_cores, "gpu": 1},
      resources_per_trial={"cpu": 3, "gpu": 1},
      mode = config["mode"],
    )

    best_primitive_config = primitive_analysis.get_best_config(metric=config['metric'], mode=config["mode"])
    best_primitive_list =  primitive_analysis.dataframe(metric=config['metric'], mode=config["mode"])

    hyperparm_search_space,is_hyperparam = self.hyperparam_searchspace(search_space,best_primitive_list,best_primitive_config)
    
    if config["ignore_hyperparameters"] is True or is_hyperparam is False:
      return self.get_search_result(primitive_analysis,config)

    print("Best primitive config: ", best_primitive_config )
    # hyperparam_searcher = self.set_search_algorithm(config["searching_algorithm"])
    # from ray.tune.suggest.hyperopt import HyperOptSearch
    hyperparam_analysis = ray.tune.run(
      self._evaluate,
      metric = config['metric'],
      config = hyperparm_search_space,
      num_samples = config["num_samples"],
      # resources_per_trial = {"cpu": num_cores, "gpu": 1},
      resources_per_trial={"cpu": 3, "gpu": 1},
      mode = config["mode"],
      # search_alg=hyperparam_searcher
    )

    # best_config = hyperparam_analysis.get_best_config(metric='F_beta', mode=config["mode"])

    return [self.get_search_result(primitive_analysis,config),self.get_search_result(hyperparam_analysis,config)]





  def get_search_result(self, analysis,config):
    """
    Get best config and best config pipeline id

    Parameters
    ----------
    analysis: ResultGrid object
      An ResultGrid object which has methods you can use for analyzing your training.

    Returns
    -------
      result: 
        A dataframe that contains search result: scores[name, score], pipeline_id, pipeline_config, search_history
        

    """
    result = {}
    best_config = analysis.get_best_config(metric=config['metric'], mode=config['mode'])
    result['best_config'] = best_config
    # df = analysis.results_df
    df = analysis.dataframe(metric=config['metric'], mode=config['mode'])
    # print(df)
    df.to_csv('out.csv')
    result['search_history'] = df
    result['best_pipeline_id'] = self.find_best_pipeline(best_config, df)
    # result['scores'] = best_result[config['metric']]
    return result


  # Initialize the trainable module
  def _evaluate(self, search_space):
    """
    Trainable module of Ray Tune. An object that need to pass into a Tune run(Tuner in Ray 2.0+).

    Define the hyperparameters you want to tune in a search space
    and pass them into a trainable that specifies the objective you want to tune.

    Parameters
    ----------
    search_space:
      A search space transform from a json file user designed
      defines valid values for your hyperparameters and can specify how these values are sampled.

    Returns
    -------

    """

    # build pipeline
    pipeline = build_pipeline(search_space)

    # print('='*50)
    # print(pipeline.to_json_structure())
    # print('='*50)
    
    
    # Train the pipeline with the specified metric and dataset.
    # And get the result
    fitted_pipeline = fit_pipeline(self.dataset, pipeline, self.metric)

    # Save fitted pipeline id
    fitted_pipeline_id = save_fitted_pipeline(fitted_pipeline[0])

    # Add fitted_pipeline_id to fitted_pipeline_list
    self.stats.append_fitted_pipeline_id.remote(fitted_pipeline_id)

    # Add d3m.metadata.pipeline.Pipeline object to pipeline_description_list
    self.stats.append_pipeline_description.remote(pipeline)


    df = self.dataframe

    y_true = df['anomaly']
    # print(pipeline_result.__dict__)
    y_pred = fitted_pipeline[1].exposed_outputs['outputs.0']['anomaly']
    # print(y_pred,type(y_pred))
    # self.stats.append_score.remote(score)

    eval_metric = get_evaluate_metric(y_true,y_pred, self.beta, self.metric)

    # ray.tune.report(score = score * 100)
    # ray.tune.report(accuracy=1)

    from random import seed
    from random import random
    from datetime import datetime
    seed(datetime.now())

    # import random
    # from datetime import datetime
    # temp = random.seed(datetime.now())

    temp = random()

    yield eval_metric



  def find_best_pipeline(self, best_config, results_dataframe):
    """
    Output pipeline ID that have the best config

    Parameters
    ----------
    best_config
    results_dataframe

    Returns
    -------

    """
    # print(results_dataframe['config/detection_algorithm'][0][0])
    # print(best_config)
    for key, value in best_config.items():
      # print(key)
      # print(results_dataframe['config/' + str(key)])
      # print(value)
      # print()
      results_dataframe = results_dataframe.loc[results_dataframe['config/' + str(key)].apply(lambda x: x == value)]
      # print(results_dataframe)
      
    return ray.get(self.stats.get_fitted_pipeline_list.remote())[results_dataframe.index[0]]


  def json_to_primitive_searchspace(self,json,is_exhaustive = 1):
    """

    Parameters
    ----------
    json: dict
      user defined search space in json format
    is_exhaustive: Boolean
      If True, use exhaustive search all the combination. If False, use simple search space.
      [[first],[first,second],[first,second,third],...]

    Returns
    -------
      primitive_searchspace: dict
        Ray.Tune search space
    """
    primitive_searchspace = {}
    if is_exhaustive:
      for key,value in json.items():
        # only support one detection algorithm per pipeline
        if key == 'detection_algorithm' and value:
          primitive_searchspace['detection_algorithm'] = ray.tune.grid_search([[[detection_algo,]] for detection_algo in value.keys()])
          continue
        primitive_searchspace[key] = self.exhaustive_searchspace(list(value.keys()))

    else:
      for key,value in json.items():
        # only support one detection algorithm per pipeline
        if key == 'detection_algorithm' and value:
          primitive_searchspace['detection_algorithm'] = ray.tune.grid_search([[[detection_algo,]] for detection_algo in value.keys()])
          continue
        primitive_searchspace[key] = self.simple_searchspace(list(value.keys()))

    return primitive_searchspace

  def exhaustive_searchspace(self,primitive_list):

    search_space_list = []
    primitive_combination = []

    def backtracking(list, start):
      if start != 0:
        search_space_list.append(primitive_combination[:])
      if start == len(list):
        return

      for i in range(start, len(list)):
        primitive_combination.append([list[i],])
        # print(primitive_combination, start)
        backtracking(list, i + 1)
        primitive_combination.pop()

    backtracking(primitive_list, 0)
    return ray.tune.grid_search(search_space_list)


  def simple_searchspace(self,list):
    res = []
    path = []

    for i in list:
      path.append([i,])
      res.append(path[:])
    # use test search space just change res to path
    return ray.tune.grid_search(res)

  # TODO provide choices of search algorithms
  def hyperparam_searchspace(self,json_data,primitive_df,primitive_config):
    list_hyperparam = ['comp_hiddens', 'est_hiddens', 'hidden_neurons']
    config={}
    is_hyperparam = False # detect if there are combinations of hypermeter if not just skip hypermeter searching
    for module, primitive_list in primitive_config.items():
        lists=[]
        for primitive in primitive_list:
          primitives = list(primitive)
          hyperparam_list = {}
          for hyperparam in json_data[module][primitive[0]]:
            # print(module,primitive,hyperparam,json_data[module][primitive[0]][hyperparam])
            data = json_data[module][primitive[0]][hyperparam]
            
            # hyperparameter in list_hyperparam is list
            if hyperparam in list_hyperparam and isinstance(data[0],list) is False:
              continue
            # hyperparams is not a list, don't need to search
            if isinstance(data,list) is False:
              continue
            
            # TODO define search space using search space API, need to be customized according to search algorithm
            data = ray.tune.grid_search(data)
            hyperparam_list[hyperparam] = data
            is_hyperparam = True
          primitives.append(hyperparam_list)
        lists.append(primitives)
        config[module] = lists
    # print("Hyperparameter Search Space: ")
    # print(config)

    return config,is_hyperparam





