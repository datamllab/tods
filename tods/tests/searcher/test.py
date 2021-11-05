import unittest
from d3m.metadata import base as metadata_base
from tods.searcher.searcher import RaySearcher, datapath_to_dataset, json_to_searchspace
from d3m import container, utils
import argparse
import os
import ray
from tods import generate_dataset, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline, compare_two_pipeline_description

import pandas as pd

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys

pipeline_description_list = []

pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                                          data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)

step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
pipeline_description.add_step(step_3)

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

step_5= PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

descrip = pipeline_description.to_json()

pipeline_description_list.append(descrip)
# ---------------------------------------------------------------------------------
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                                          data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)

step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
pipeline_description.add_step(step_3)

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

step_5= PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

step_6= PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.telemanom'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')

descrip = pipeline_description.to_json()

pipeline_description_list.append(descrip)


# scores = ray.get(self.stats.get_scores.remote())
# pipess = ray.get(self.stats.get_pipeline_description_list.remote())

class SeacherTest(unittest.TestCase):
  def test_searcher(self):
    self.data_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
    self.target_index = 6

    self.data = datapath_to_dataset(self.data_path, self.target_index)


    self.searcher = RaySearcher(self.data, 'F1_MACRO')

    self.config = {"searching_algorithm": 'hyperopt',
    "num_samples": 6,
    "mode": 'max'
    }

    from ray import tune
    self.search_space = {
    "timeseries_processing": tune.choice(["time_series_seasonality_trend_decomposition"]),
    "feature_analysis": tune.choice(["statistical_maximum"]),
    "detection_algorithm": tune.choice(["pyod_ae", "pyod_ae telemanom"])
    }

    best_config, best_pipeline_id = self.searcher.search(search_space=self.search_space, config=self.config)

    self.stats = self.searcher.stats

    self.assertIsInstance(self.searcher, RaySearcher)
    self.scores = ray.get(self.stats.get_scores.remote())
    self.pipess = ray.get(self.stats.get_pipeline_description_list.remote())

    import json
    for i in range(len(pipeline_description_list)):
      pipeline_description_list[i] = json.loads(pipeline_description_list[i])
      pipeline_description_list[i]['id'] = ''
      pipeline_description_list[i]['created'] = ''
      pipeline_description_list[i]['digest'] = ''
      pipeline_description_list[i]['schema'] = ''

    for i in range(len(self.pipess)):
      self.pipess[i] = json.loads(self.pipess[i])
      self.pipess[i]['id'] = ''
      self.pipess[i]['created'] = ''
      self.pipess[i]['digest'] = ''
      self.pipess[i]['schema'] = ''

    res = True
    for i in self.pipess:
      if i in pipeline_description_list:
        continue
      else:
        res = False

    self.assertTrue(res)

  # def test_pipe(self):
  #   import json
  #   for i in range(len(self.pipeline_description_list)):
  #     self.pipeline_description_list[i] = json.loads(self.pipeline_description_list[i])
  #     self.pipeline_description_list[i]['id'] = ''
  #     self.pipeline_description_list[i]['created'] = ''
  #     self.pipeline_description_list[i]['digest'] = ''
  #     self.pipeline_description_list[i]['schema'] = ''

  #   for i in range(len(self.pipess)):
  #     self.pipess[i] = json.loads(self.pipess[i])
  #     self.pipess[i]['id'] = ''
  #     self.pipess[i]['created'] = ''
  #     self.pipess[i]['digest'] = ''
  #     self.pipess[i]['schema'] = ''

  #   res = True
  #   for i in self.pipess:
  #     if i in self.pipeline_description_list:
  #       continue
  #     else:
  #       res = False

  #   self.assertTrue(res)


if __name__ == '__main__':
  unittest.main()