import unittest
from d3m.metadata import base as metadata_base
from tods.searcher.searcher import RaySearcher, datapath_to_dataset, json_to_searchspace
from d3m import container, utils
import argparse
import os
import ray
from tods import generate_dataset, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline, compare_two_pipeline_description

import pandas as pd
import numpy as np

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys
import d3m
class LoadSaveTest(unittest.TestCase):
  def test_load_save(self):
    table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
    df = pd.read_csv(table_path)
    self.dataset = generate_dataset(df, 6)

    # Creating pipeline
    self.pipeline_description = Pipeline()
    self.pipeline_description.add_input(name='inputs')

    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    self.pipeline_description.add_step(step_0)

    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    self.pipeline_description.add_step(step_1)

    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                    data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    self.pipeline_description.add_step(step_2)

    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                  data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    self.pipeline_description.add_step(step_3)

    attributes = 'steps.2.produce'
    targets = 'steps.3.produce'

    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
    step_4.add_output('produce')
    self.pipeline_description.add_step(step_4)

    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
    step_5.add_hyperparameter(name='contamination', argument_type=ArgumentType.VALUE, data=0.4)
    step_5.add_hyperparameter(name='dropout_rate', argument_type=ArgumentType.VALUE, data=0.6)
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_output('produce')
    self.pipeline_description.add_step(step_5)

    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.telemanom'))
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_6.add_output('produce')
    self.pipeline_description.add_step(step_6)

    step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
    step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
    step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_7.add_output('produce')
    self.pipeline_description.add_step(step_7)

    # step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
    # step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    # step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    # step_6.add_output('produce')
    # pipeline_description.add_step(step_6)

    # Final Output
    self.pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')
    self.pipeline_description.to_json()


    LoadSaveTest.fitted_pipeline = fit_pipeline(self.dataset, self.pipeline_description, 'F1_MACRO')
    print(type(self.fitted_pipeline['runtime']))
    print(type(self.fitted_pipeline['dataset_metadata']))
    
    self.assertIsInstance(self.fitted_pipeline, dict)
    self.assertIsInstance(self.fitted_pipeline['runtime'], d3m.runtime.Runtime)
    self.assertIsInstance(self.fitted_pipeline['dataset_metadata'], d3m.metadata.base.DataMetadata)

    LoadSaveTest.fitted_pipeline_id = save_fitted_pipeline(LoadSaveTest.fitted_pipeline)

    LoadSaveTest.loaded_pipeline = load_fitted_pipeline(LoadSaveTest.fitted_pipeline_id)

    # print(LoadSaveTest.fitted_pipeline['runtime'].steps_state)
    print(LoadSaveTest.loaded_pipeline['runtime'].steps_state[6]['clf_'])
    steps_state = LoadSaveTest.loaded_pipeline['runtime'].steps_state
    autoencoder_model = steps_state[5]['clf_']
    telemanom = steps_state[6]['clf_']
    # print()

    self.assertEqual(steps_state[0], None)
    self.assertEqual(steps_state[1], None)
    self.assertEqual(steps_state[2], None)
    self.assertEqual(steps_state[3], None)
    self.assertEqual(steps_state[4], None)
    self.assertEqual(steps_state[7], None)

    self.assertEqual(autoencoder_model.batch_size, 32)
    self.assertEqual(autoencoder_model.contamination, 0.4)
    self.assertEqual(autoencoder_model.dropout_rate, 0.6)
    self.assertEqual(autoencoder_model.epochs, 20)
    self.assertEqual(autoencoder_model.hidden_activation, 'relu')
    self.assertEqual(autoencoder_model.hidden_neurons, [1, 4, 1])
    self.assertEqual(autoencoder_model.l2_regularizer, 0.1)
    self.assertEqual(autoencoder_model.optimizer, 'adam')
    self.assertEqual(autoencoder_model.output_activation, 'sigmoid')
    self.assertEqual(autoencoder_model.preprocessing, True)
    self.assertEqual(autoencoder_model.random_state, None)
    self.assertEqual(autoencoder_model.validation_size, 0.1)
    self.assertEqual(autoencoder_model.verbose, 1)

    self.assertEqual(telemanom._batch_size, 70)
    self.assertEqual(telemanom.contamination, 0.1)
    self.assertEqual(telemanom._smoothin_perc, 0.05)
    self.assertEqual(telemanom._window_size, 100)
    self.assertEqual(telemanom._error_buffer, 50)
    self.assertEqual(telemanom._batch_size, 70)
    self.assertEqual(telemanom._loss_metric, 'mean_squared_error')
    self.assertEqual(telemanom._layers, [10,10])
    self.assertEqual(telemanom._epochs, 1)
    self.assertEqual(telemanom._patience, 10)
    self.assertEqual(telemanom._min_delta, 0.0003)
    self.assertEqual(telemanom._l_s, 100)
    self.assertEqual(telemanom._n_predictions, 10)
    self.assertEqual(telemanom._p, 0.05)






if __name__ == '__main__':
  unittest.main()
