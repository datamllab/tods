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
  def __init__(self, dataset, metric,path,beta):
    ray.init(local_mode=True, ignore_reinit_error=True)
    self.dataset = dataset
    self.metric = metric
    self.stats = GlobalStats.remote()
    self.path=path
    self.beta=beta


  def search(self,search_space, config):
    """
    Launch a search process to find a suitable pipeline

    Parameters
    ----------
    search_space:
        A search space transform from a json file user designed
        defines valid values for your hyperparameters and can specify how these values are sampled.
    config:
        An object containing parameters defined by the user to set how the searcher will run

    Returns
    -------
      best_config
      best_config_pipeline_id
    """




    primitive_search_space = self.json_to_primitive_searchspace(search_space, is_exhaustive=config["use_all_combinations"])

    primitive_searcher = self.set_search_algorithm(config["searching_algorithm"])


    # set the number of CPU cores
    import multiprocessing
    num_cores =  multiprocessing.cpu_count()

    # Launch searcher
    primitive_analysis = ray.tune.run(
      self._evaluate,
      metric = "F_beta",
      config = primitive_search_space,
      num_samples = 1,
      # resources_per_trial = {"cpu": num_cores, "gpu": 1},
      resources_per_trial={"cpu": 3, "gpu": 1},
      mode = config["mode"],
    )

    best_primitive_config = primitive_analysis.get_best_config(metric='F_beta', mode=config["mode"])
    best_primitive_list =  primitive_analysis.dataframe(metric="F_beta", mode=config["mode"])

    if config["ignore_hyperparameters"] is True:
      return self.get_best_config(primitive_analysis)

    print("Best primitive config: ", best_primitive_config )
    hyperparm_search_space = self.hyperparam_searchspace(search_space,best_primitive_list,best_primitive_config)

    hyperparam_searcher = self.set_search_algorithm(config["searching_algorithm"])
    # from ray.tune.suggest.hyperopt import HyperOptSearch
    hyperparam_analysis = ray.tune.run(
      self._evaluate,
      metric = "F_beta",
      config = hyperparm_search_space,
      num_samples = config["num_samples"],
      # resources_per_trial = {"cpu": num_cores, "gpu": 1},
      resources_per_trial={"cpu": 3, "gpu": 1},
      mode = config["mode"],
      # search_alg=HyperOptSearch()
    )

    # best_config = hyperparam_analysis.get_best_config(metric='F_beta', mode=config["mode"])

    best_config, best_config_pipeline_id = self.get_best_config(hyperparam_analysis)

    return best_config, best_config_pipeline_id


  def set_search_algorithm(self, algorithm):
    """
    Determine which searcher to choose based on user needs

    Parameters
    ----------
    algorithm: str
      Name of the desired search algorithm to use

    Returns
    -------
    searcher
    """

    if algorithm == "random":
      from ray.tune.suggest.basic_variant import BasicVariantGenerator
      searcher = BasicVariantGenerator()  # Random/Grid Searcher
    elif algorithm == "hyperopt":
      from ray.tune.suggest.hyperopt import HyperOptSearch
      searcher = HyperOptSearch(max_concurrent=2, metric="RECALL")  # HyperOpt Searcher
    elif algorithm == "zoopt":
      zoopt_search_config = {
        "parallel_num": 64,  # how many workers to parallel
      }
      from ray.tune.suggest.zoopt import ZOOptSearch
      searcher = ZOOptSearch(budget=20, **zoopt_search_config)
    elif algorithm == "skopt":
      from ray.tune.suggest.skopt import SkOptSearch
      searcher = SkOptSearch()
    elif algorithm == "nevergrad":
      import nevergrad as ng
      from ray.tune.suggest.nevergrad import NevergradSearch
      searcher = NevergradSearch(
        optimizer=ng.optimizers.OnePlusOne)
    else:
      raise ValueError("Searching algorithm not supported.")
    return searcher


  def get_best_config(self, analysis):
    """
    Get best config and best config pipeline id

    Parameters
    ----------
    analysis: ResultGrid object
      An ResultGrid object which has methods you can use for analyzing your training.

    Returns
    -------
      best_config:
        A config which achieves the best performance
        obtained by searcher analysis
      best_config_pipeline_id:


    """

    best_config = analysis.get_best_config(metric='F_beta', mode="max")
    # df = analysis.results_df
    df = analysis.dataframe(metric="F_beta", mode="max")
    # print(df)
    df.to_csv('out.csv')
    best_config_pipeline_id = self.find_best_pipeline_id(best_config, df)
    return best_config, best_config_pipeline_id


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

    # Train the pipeline with the specified metric and dataset.
    # And get the result
    fitted_pipeline,pipeline_result = fit_pipeline(self.dataset, pipeline, self.metric)

    # Save fitted pipeline id
    fitted_pipeline_id = save_fitted_pipeline(fitted_pipeline)

    # Add fitted_pipeline_id to fitted_pipeline_list
    self.stats.append_fitted_pipeline_id.remote(fitted_pipeline_id)

    # Add d3m.metadata.pipeline.Pipeline object to pipeline_description_list
    self.stats.append_pipeline_description.remote(pipeline)


    df = pd.read_csv(self.path)

    y_true = df['anomaly']
    y_pred = pipeline_result.exposed_outputs['outputs.0']['anomaly']
    # self.stats.append_score.remote(score)

    eval_metric = get_evaluate_metric(y_true,y_pred,self.beta,self.metric)

    # ray.tune.report(score = score * 100)
    # ray.tune.report(accuracy=1)s

    from random import seed
    from random import random
    from datetime import datetime
    seed(datetime.now())

    # import random
    # from datetime import datetime
    # temp = random.seed(datetime.now())

    temp = random()

    yield eval_metric


  def build_pipeline(self, search_space):
    """
    Build an outlier detection system
    Args:
      search_space:
        A search space transform from a json file user designed
        defines valid values for your hyperparameters and can specify how these values are sampled.

    """
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
        # setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.' + primitive_map[timeseries_processing_list[x]] + '.' +  timeseries_processing_list[x])))
        # this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.' + primitive_map[timeseries_processing_list[x]] + '.' +  timeseries_processing_list[x]))



        setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.' + timeseries_processing_list[x])))
        this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.' +  timeseries_processing_list[x]))

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
      setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.' + detection_algorithm_list[x])))
      this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.' + detection_algorithm_list[x]))
      # print(this.name.metadata['hyperparams_to_tune'])
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
      setattr(this, name, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions')))
      this.name = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))

      this.name.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.' + str(counter - 1) + '.produce')
      this.name.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
      this.name.add_output('produce')
      pipeline_description.add_step(this.name)
      counter += 1




    pipeline_description.add_output(name='output predictions', data_reference='steps.' + str(counter - 1) + '.produce')
    data = pipeline_description.to_json()

    # input()
    return pipeline_description



  def clearer_best_config(self, best_config):
    """
    Output the best config in a clearer format

    Parameters
    ----------
    best_config:
        A config which achieves the best performance
        obtained by searcher analysis

    Returns
    -------
    None
    """

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

  def find_best_pipeline_id(self, best_config, results_dataframe):
    """
    Output pipeline ID that have the best config

    Parameters
    ----------
    best_config
    results_dataframe

    Returns
    -------

    """

    for key, value in best_config.items():
        results_dataframe = results_dataframe.loc[results_dataframe['config/' + str(key)].apply(lambda x: x == value)]

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
        if key == 'detection_algorithm' and value:
          # only support one detection algorithm per pipeline
          primitive_searchspace['detection_algorithm'] = ray.tune.grid_search([[(detection_algo,)] for detection_algo in value.keys()])
          continue
        primitive_searchspace[key] = self.exhaustive_searchspace(list(value.keys()))

    else:
      for key,value in json.items():
        if key == 'detection_algorithm' and value:
          # only support one detection algorithm per pipeline
          primitive_searchspace['detection_algorithm'] = ray.tune.grid_search([[(detection_algo,)] for detection_algo in value.keys()])
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
        primitive_combination.append((list[i],))
        print(primitive_combination, start)
        backtracking(list, i + 1)
        primitive_combination.pop()

    backtracking(primitive_list, 0)
    return ray.tune.grid_search(search_space_list)


  def simple_searchspace(self,list):
    res = []
    path = []

    for i in list:
      path.append((i,))
      res.append(path[:])

    return ray.tune.grid_search(res)


  def hyperparam_searchspace(self,json_data,primitive_df,primitive_config):
    list_hyperparam = ['comp_hiddens', 'est_hiddens', 'hidden_neurons']
    config={}
    for module, primitive_list in primitive_config.items():
        lists=[]
        for primitive in primitive_list:
          primitives = list(primitive)
          hyperparam_list = {}
          for hyperparam in json_data[module][primitive[0]]:
            print(module,primitive,hyperparam,json_data[module][primitive[0]][hyperparam])
            data = json_data[module][primitive[0]][hyperparam]
            if hyperparam in list_hyperparam:
              if isinstance(data[0],list) is False:
                continue
            if isinstance(data,list) is False:
              continue
            data = ray.tune.grid_search(data)
            hyperparam_list[hyperparam] = data
          primitives.append(hyperparam_list)
        lists.append(primitives)
        config[module] = lists

    print("Hyperparam Search Space: ")
    print(config)

    return config





