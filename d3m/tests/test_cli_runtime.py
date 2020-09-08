import contextlib
import json
import gzip
import io
import logging
import os.path
import pickle
import random
import shutil
import sys
import tempfile
import traceback
import unittest

import pandas

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')
sys.path.insert(0, TEST_PRIMITIVES_DIR)

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.construct_predictions import ConstructPredictionsPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.no_split import NoSplitDatasetSplitPrimitive
from common_primitives.random_forest import RandomForestClassifierPrimitive
from common_primitives.train_score_split import TrainScoreDatasetSplitPrimitive


from test_primitives.random_classifier import RandomClassifierPrimitive
from test_primitives.fake_score import FakeScorePrimitive

from d3m import cli, index, runtime, utils
from d3m.container import dataset as dataset_module
from d3m.contrib.primitives.compute_scores import ComputeScoresPrimitive
from d3m.metadata import base as metadata_base, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem as problem_module

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PROBLEM_DIR = os.path.join(TEST_DATA_DIR, 'problems')
DATASET_DIR = os.path.join(TEST_DATA_DIR, 'datasets')
PIPELINE_DIR = os.path.join(TEST_DATA_DIR, 'pipelines')


class TestCLIRuntime(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @classmethod
    def setUpClass(cls):
        to_register = {
            'd3m.primitives.data_transformation.dataset_to_dataframe.Common': DatasetToDataFramePrimitive,
            'd3m.primitives.classification.random_forest.Common': RandomForestClassifierPrimitive,
            'd3m.primitives.classification.random_classifier.Test': RandomClassifierPrimitive,
            'd3m.primitives.data_transformation.column_parser.Common': ColumnParserPrimitive,
            'd3m.primitives.data_transformation.construct_predictions.Common': ConstructPredictionsPrimitive,
            'd3m.primitives.evaluation.no_split_dataset_split.Common': NoSplitDatasetSplitPrimitive,
            'd3m.primitives.evaluation.compute_scores.Test': FakeScorePrimitive,
            'd3m.primitives.evaluation.train_score_dataset_split.Common': TrainScoreDatasetSplitPrimitive,
            # We do not have to load this primitive, but loading it here prevents the package from loading all primitives.
            'd3m.primitives.evaluation.compute_scores.Core': ComputeScoresPrimitive,
        }

        # To hide any logging or stdout output.
        with utils.silence():
            for python_path, primitive in to_register.items():
                index.register_primitive(python_path, primitive)

    def _call_cli_runtime(self, arg):
        logger = logging.getLogger('d3m.runtime')
        with utils.silence():
            with self.assertLogs(logger=logger) as cm:
                # So that at least one message is logged.
                logger.warning("Debugging.")
                cli.main(arg)
        # We skip our "debugging" message.
        return cm.records[1:]

    def _call_cli_runtime_without_fail(self, arg):
        try:
            return self._call_cli_runtime(arg)
        except Exception as e:
            self.fail(traceback.format_exc())

    def _assert_valid_saved_pipeline_runs(self, pipeline_run_save_path):
        with open(pipeline_run_save_path, 'r') as f:
            for pipeline_run_dict in list(utils.yaml_load_all(f)):
                try:
                    pipeline_run_module.validate_pipeline_run(pipeline_run_dict)
                except Exception as e:
                    self.fail(traceback.format_exc())

    def _validate_previous_pipeline_run_ids(self, pipeline_run_save_path):
        ids = set()
        prev_ids = set()
        with open(pipeline_run_save_path, 'r') as f:
            for pipeline_run_dict in list(utils.yaml_load_all(f)):
                ids.add(pipeline_run_dict['id'])
                if 'previous_pipeline_run' in pipeline_run_dict:
                    prev_ids.add(pipeline_run_dict['previous_pipeline_run']['id'])
        self.assertTrue(
            prev_ids.issubset(ids),
            'Some previous pipeline run ids {} are not in the set of pipeline run ids {}'.format(prev_ids, ids)
        )

    def test_fit_multi_input(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--problem',
            os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'multi-input-test.json'),
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._assert_standard_output_metadata()

    def test_fit_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-pipeline')
        output_csv_path = os.path.join(self.test_dir, 'output.csv')
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'multi-input-test.json'),
            '--save',
            fitted_pipeline_path,
            '--expose-produced-outputs',
            self.test_dir,
            '--output',
            output_csv_path,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'fitted-pipeline',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11225, outputs_path='outputs.0/data.csv')
        self._assert_prediction_sum(prediction_sum=11225, outputs_path='output.csv')

    def test_produce_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-no-problem-pipeline')
        output_csv_path = os.path.join(self.test_dir, 'output.csv')
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'multi-input-test.json'),
            '--save',
            fitted_pipeline_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        arg = [
            '',
            'runtime',
            'produce',
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--output',
            output_csv_path,
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'fitted-no-problem-pipeline',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11008, outputs_path='outputs.0/data.csv')
        self._assert_prediction_sum(prediction_sum=11008, outputs_path='output.csv')

    def test_fit_produce_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        output_csv_path = os.path.join(self.test_dir, 'output.csv')
        arg = [
            '',
            'runtime',
            'fit-produce',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'multi-input-test.json'),
            '--output',
            output_csv_path,
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)
        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11008, outputs_path='outputs.0/data.csv')
        self._assert_prediction_sum(prediction_sum=11008, outputs_path='output.csv')

    def test_nonstandard_fit_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-pipeline')
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'semi-standard-pipeline.json'),
            '--save',
            fitted_pipeline_path,
            '--expose-produced-outputs',
            self.test_dir,
            '--not-standard-pipeline',
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'fitted-pipeline',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'outputs.1/data.csv',
            'outputs.1/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=10710, outputs_path='outputs.0/data.csv')
        self._assert_nonstandard_output(outputs_name='outputs.1')

    def test_nonstandard_produce_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-pipeline')
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'semi-standard-pipeline.json'),
            '--save',
            fitted_pipeline_path,
            '--not-standard-pipeline'
        ]
        self._call_cli_runtime_without_fail(arg)

        arg = [
            '',
            'runtime',
            'produce',
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'fitted-pipeline',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'outputs.1/data.csv',
            'outputs.1/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json'
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=12106, outputs_path='outputs.0/data.csv')
        self._assert_nonstandard_output(outputs_name='outputs.1')

    def test_nonstandard_fit_produce_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit-produce',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'semi-standard-pipeline.json'),
            '--expose-produced-outputs',
            self.test_dir,
            '--not-standard-pipeline',
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'outputs.1/data.csv',
            'outputs.1/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)
        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=12106, outputs_path='outputs.0/data.csv')
        self._assert_nonstandard_output(outputs_name='outputs.1')

    def test_fit_produce_multi_input(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit-produce',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--problem',
            os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'multi-input-test.json'),
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertEqual(utils.list_files(self.test_dir), [
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json',
        ])

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)
        self._assert_standard_output_metadata()
        self._assert_prediction_sum(prediction_sum=11008, outputs_path='outputs.0/data.csv')

    def test_fit_score(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit-score',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--problem',
            os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--score-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-forest-classifier.yml'),
            '--scores',
            os.path.join(self.test_dir, 'scores.csv'),
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        dataframe = pandas.read_csv(os.path.join(self.test_dir, 'scores.csv'))
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])

    def test_fit_score_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit-score',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--score-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-classifier.yml'),
            '--scoring-pipeline',
            os.path.join(PIPELINE_DIR, 'fake_compute_score.yml'),
            # this argument has no effect
            '--metric',
            'F1_MACRO',
            '--metric',
            'ACCURACY',
            '--scores',
            os.path.join(self.test_dir, 'scores.csv'),
            '-O',
            pipeline_run_save_path,
        ]
        logging_records = self._call_cli_runtime_without_fail(arg)

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        dataframe = pandas.read_csv(os.path.join(self.test_dir, 'scores.csv'))
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])

    @staticmethod
    def _get_iris_dataset_path():
        return os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json')

    @staticmethod
    def _get_iris_problem_path():
        return os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json')

    @staticmethod
    def _get_random_forest_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'random-forest-classifier.yml')

    @staticmethod
    def _get_no_split_data_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'data-preparation-no-split.yml')

    @staticmethod
    def _get_train_test_split_data_pipeline_path():
        return os.path.join(PIPELINE_DIR, 'data-preparation-train-test-split.yml')

    def _get_pipeline_run_save_path(self):
        return os.path.join(self.test_dir, 'pipeline_run.yml')

    def _get_predictions_path(self):
        return os.path.join(self.test_dir, 'predictions.csv')

    def _get_scores_path(self):
        return os.path.join(self.test_dir, 'scores.csv')

    def _get_pipeline_rerun_save_path(self):
        return os.path.join(self.test_dir, 'pipeline_rerun.yml')

    def _get_rescores_path(self):
        return os.path.join(self.test_dir, 'rescores.csv')

    def _fit_iris_random_forest(
        self, *, predictions_path=None, fitted_pipeline_path=None, pipeline_run_save_path=None
    ):
        if pipeline_run_save_path is None:
            pipeline_run_save_path = self._get_pipeline_run_save_path()
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            self._get_iris_dataset_path(),
            '--problem',
            self._get_iris_problem_path(),
            '--pipeline',
            self._get_random_forest_pipeline_path(),
            '-O',
            pipeline_run_save_path
        ]
        if predictions_path is not None:
            arg.append('--output')
            arg.append(predictions_path)
        if fitted_pipeline_path is not None:
            arg.append('--save')
            arg.append(fitted_pipeline_path)

        self._call_cli_runtime_without_fail(arg)

    def _fit_iris_random_classifier_without_problem(self, *, fitted_pipeline_path):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-classifier.yml'),
            '-O',
            pipeline_run_save_path
        ]
        if fitted_pipeline_path is not None:
            arg.append('--save')
            arg.append(fitted_pipeline_path)

        self._call_cli_runtime_without_fail(arg)

    def test_fit(self):
        pipeline_run_save_path = self._get_pipeline_run_save_path()
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-pipeline')
        self._fit_iris_random_forest(
            fitted_pipeline_path=fitted_pipeline_path, pipeline_run_save_path=pipeline_run_save_path
        )

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self.assertTrue(os.path.isfile(fitted_pipeline_path))
        self.assertTrue(os.path.isfile(pipeline_run_save_path))

    def test_evaluate(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'evaluate',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--problem',
            os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-forest-classifier.yml'),
            '--data-pipeline',
            os.path.join(PIPELINE_DIR, 'data-preparation-no-split.yml'),
            '--scores',
            scores_path,
            '--metric',
            'ACCURACY',
            '--metric',
            'F1_MACRO',
            '-O',
            pipeline_run_save_path
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0, 0], ['F1_MACRO', 1.0, 1.0, 0, 0]])

    def test_evaluate_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'evaluate',
            '--input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-classifier.yml'),
            '--data-pipeline',
            os.path.join(PIPELINE_DIR, 'data-preparation-no-split.yml'),
            '--scoring-pipeline',
            os.path.join(PIPELINE_DIR, 'fake_compute_score.yml'),
            # this argument has no effect
            '--metric',
            'ACCURACY',
            '--scores',
            scores_path,
            '-O',
            pipeline_run_save_path
        ]
        logging_records = self._call_cli_runtime_without_fail(arg)

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0, 0]])

    def test_score(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'iris-pipeline')
        self._fit_iris_random_forest(fitted_pipeline_path=fitted_pipeline_path)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'score',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--score-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--scores',
            scores_path,
            '--metric',
            'F1_MACRO',
            '--metric',
            'ACCURACY',
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['F1_MACRO', 1.0, 1.0, 0], ['ACCURACY', 1.0, 1.0, 0]])

    def test_score_without_problem_without_metric(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'iris-pipeline')
        self._fit_iris_random_classifier_without_problem(fitted_pipeline_path=fitted_pipeline_path)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'score',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--score-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--scoring-pipeline',
            os.path.join(PIPELINE_DIR, 'fake_compute_score.yml'),
            '--scores',
            scores_path,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])

    def test_score_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'iris-pipeline')
        self._fit_iris_random_classifier_without_problem(fitted_pipeline_path=fitted_pipeline_path)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'score',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--score-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--scoring-pipeline',
            os.path.join(PIPELINE_DIR, 'fake_compute_score.yml'),
            # this argument has no effect
            '--metric',
            'ACCURACY',
            '--scores',
            scores_path,
            '-O',
            pipeline_run_save_path,
        ]
        logging_records = self._call_cli_runtime_without_fail(arg)

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0]])

    def test_produce(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'iris-pipeline')
        self._fit_iris_random_forest(fitted_pipeline_path=fitted_pipeline_path)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        arg = [
            '',
            'runtime',
            'produce',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--test-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

    def test_score_predictions(self):
        predictions_path = os.path.join(self.test_dir, 'predictions.csv')
        self._fit_iris_random_forest(predictions_path=predictions_path)
        self.assertTrue(os.path.isfile(predictions_path))

        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'score-predictions',
            '--score-input',
            os.path.join(DATASET_DIR, 'iris_dataset_1/datasetDoc.json'),
            '--problem',
            os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'),
            '--predictions',
            predictions_path,
            '--metric',
            'ACCURACY',
            '--metric',
            'F1_MACRO',
            '--scores',
            scores_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0], ['F1_MACRO', 1.0, 1.0]])

    def test_sklearn_dataset_fit_produce(self):
        self._create_sklearn_iris_problem_doc()

        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        arg = [
            '',
            'runtime',
            'fit-produce',
            '--input',
            'sklearn://iris',
            '--input',
            'sklearn://iris',
            '--problem',
            os.path.join(self.test_dir, 'problemDoc.json'),
            '--test-input',
            'sklearn://iris',
            '--test-input',
            'sklearn://iris',
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'multi-input-test.json'),
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        self.assertEqual(utils.list_files(self.test_dir), [
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'problemDoc.json',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json'
        ])
        self._assert_standard_output_metadata(prediction_type='numpy.int64')
        self._assert_prediction_sum(prediction_sum=10648, outputs_path='outputs.0/data.csv')

    def test_sklearn_dataset_fit_produce_without_problem(self):
        output_csv_path = os.path.join(self.test_dir, 'output.csv')
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-pipeline')
        arg = [
            '',
            'runtime',
            'fit-produce',
            '--input',
            'sklearn://iris',
            '--test-input',
            'sklearn://iris',
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-classifier.yml'),
            '--save',
            fitted_pipeline_path,
            '--output',
            output_csv_path,
            '--expose-produced-outputs',
            self.test_dir,
            '-O',
            pipeline_run_save_path,
        ]

        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        self.assertEqual(utils.list_files(self.test_dir), [
            'fitted-pipeline',
            'output.csv',
            'outputs.0/data.csv',
            'outputs.0/metadata.json',
            'pipeline_run.yml',
            'steps.0.produce/data.csv',
            'steps.0.produce/metadata.json',
            'steps.1.produce/data.csv',
            'steps.1.produce/metadata.json',
            'steps.2.produce/data.csv',
            'steps.2.produce/metadata.json',
        ])
        self._assert_standard_output_metadata(prediction_type='numpy.int64')
        self._assert_prediction_sum(prediction_sum=10648, outputs_path='outputs.0/data.csv')
        self._assert_prediction_sum(prediction_sum=10648, outputs_path='output.csv')

    def _create_sklearn_iris_problem_doc(self):
        with open(os.path.join(PROBLEM_DIR, 'iris_problem_1/problemDoc.json'), 'r', encoding='utf8') as problem_doc_file:
            problem_doc = json.load(problem_doc_file)

        problem_doc['inputs']['data'][0]['datasetID'] = 'sklearn://iris'

        with open(os.path.join(self.test_dir, 'problemDoc.json'), 'x', encoding='utf8') as problem_doc_file:
            json.dump(problem_doc, problem_doc_file)

    def test_sklearn_dataset_evaluate(self):
        self._create_sklearn_iris_problem_doc()

        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'evaluate',
            '--input',
            'sklearn://iris',
            '--problem',
            os.path.join(self.test_dir, 'problemDoc.json'),
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-forest-classifier.yml'),
            '--data-pipeline',
            os.path.join(PIPELINE_DIR, 'data-preparation-no-split.yml'),
            '--scores',
            scores_path,
            '--metric',
            'ACCURACY',
            '--metric',
            'F1_MACRO',
            '-O',
            pipeline_run_save_path
        ]
        self._call_cli_runtime_without_fail(arg)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0, 0], ['F1_MACRO', 1.0, 1.0, 0, 0]])

    def test_sklearn_dataset_evaluate_without_problem(self):
        pipeline_run_save_path = os.path.join(self.test_dir, 'pipeline_run.yml')
        scores_path = os.path.join(self.test_dir, 'scores.csv')
        arg = [
            '',
            'runtime',
            'evaluate',
            '--input',
            'sklearn://iris',
            '--pipeline',
            os.path.join(PIPELINE_DIR, 'random-classifier.yml'),
            '--data-pipeline',
            os.path.join(PIPELINE_DIR, 'data-preparation-no-split.yml'),
            '--scoring-pipeline',
            os.path.join(PIPELINE_DIR, 'fake_compute_score.yml'),
            # this argument has no effect
            '--metric',
            'ACCURACY',
            '--scores',
            scores_path,
            '-O',
            pipeline_run_save_path
        ]
        logging_records = self._call_cli_runtime_without_fail(arg)

        self.assertEqual(len(logging_records), 1)
        self.assertEqual(logging_records[0].msg, "Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s")

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)
        self._validate_previous_pipeline_run_ids(pipeline_run_save_path)

        dataframe = pandas.read_csv(scores_path)
        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'normalized', 'randomSeed', 'fold'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, 1.0, 0, 0]])

    def _assert_prediction_sum(self, prediction_sum, outputs_path):
        if prediction_sum is not None:
            with open(os.path.join(self.test_dir, outputs_path), 'r') as csv_file:
                self.assertEqual(sum([int(v) for v in list(csv_file)[1:]]), prediction_sum)

    def _assert_standard_output_metadata(self, outputs_name='outputs.0', prediction_type='str'):
        with open(os.path.join(self.test_dir, outputs_name, 'metadata.json'), 'r') as metadata_file:
            metadata = json.load(metadata_file)

        self.assertEqual(
            metadata,
            [
                {
                    "selector": [],
                    "metadata": {
                        "dimension": {
                            "length": 150,
                            "name": "rows",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularRow"],
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json",
                        "semantic_types": ["https://metadata.datadrivendiscovery.org/types/Table"],
                        "structural_type": "d3m.container.pandas.DataFrame",
                    },
                },
                {
                    "selector": ["__ALL_ELEMENTS__"],
                    "metadata": {
                        "dimension": {
                            "length": 1,
                            "name": "columns",
                            "semantic_types": ["https://metadata.datadrivendiscovery.org/types/TabularColumn"],
                        }
                    },
                },
                {"selector": ["__ALL_ELEMENTS__", 0],
                 "metadata": {"name": "predictions", "structural_type": prediction_type}},
            ],
        )

    def _assert_nonstandard_output(self, outputs_name='outputs.1'):
        with open(os.path.join(self.test_dir, outputs_name, 'data.csv'), 'r') as csv_file:
            output_dataframe = pandas.read_csv(csv_file, index_col=False)
            learning_dataframe = pandas.read_csv(
                os.path.join(DATASET_DIR, 'iris_dataset_1/tables/learningData.csv'), index_col=False)
            self.assertTrue(learning_dataframe.equals(output_dataframe))

        with open(os.path.join(self.test_dir, outputs_name, 'metadata.json'), 'r') as metadata_file:
            metadata = json.load(metadata_file)

        self.assertEqual(
            metadata,
            [
                {
                    "metadata": {
                        "dimension": {
                            "length": 150,
                            "name": "rows",
                            "semantic_types": [
                                "https://metadata.datadrivendiscovery.org/types/TabularRow"
                            ]
                        },
                        "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json",
                        "semantic_types": [
                            "https://metadata.datadrivendiscovery.org/types/Table"
                        ],
                        "structural_type": "d3m.container.pandas.DataFrame"
                    },
                    "selector": []
                },
                {
                    "metadata": {
                        "dimension": {
                            "length": 6,
                            "name": "columns",
                            "semantic_types": [
                                "https://metadata.datadrivendiscovery.org/types/TabularColumn"
                            ]
                        }
                    },
                    "selector": [
                        "__ALL_ELEMENTS__"
                    ]
                },
                {
                    "metadata": {
                        "name": "d3mIndex",
                        "semantic_types": [
                            "http://schema.org/Integer",
                            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        0
                    ]
                },
                {
                    "metadata": {
                        "name": "sepalLength",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        1
                    ]
                },
                {
                    "metadata": {
                        "name": "sepalWidth",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        2
                    ]
                },
                {
                    "metadata": {
                        "name": "petalLength",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        3
                    ]
                },
                {
                    "metadata": {
                        "name": "petalWidth",
                        "semantic_types": [
                            "http://schema.org/Float",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        4
                    ]
                },
                {
                    "metadata": {
                        "name": "species",
                        "semantic_types": [
                            "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                            "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                            "https://metadata.datadrivendiscovery.org/types/Attribute"
                        ],
                        "structural_type": "str"
                    },
                    "selector": [
                        "__ALL_ELEMENTS__",
                        5
                    ]
                }
            ]
        )

    def _assert_pipeline_runs_equal(self, pipeline_run_save_path1, pipeline_run_save_path2):
        with open(pipeline_run_save_path1, 'r') as f:
            pipeline_runs1 = list(utils.yaml_load_all(f))

        with open(pipeline_run_save_path2, 'r') as f:
            pipeline_runs2 = list(utils.yaml_load_all(f))

        self.assertEqual(len(pipeline_runs1), len(pipeline_runs2))

        for pipeline_run1, pipeline_run2 in zip(pipeline_runs1, pipeline_runs2):
            self.assertTrue(pipeline_run_module.PipelineRun.json_structure_equals(pipeline_run1, pipeline_run2))

    def test_pipeline_run_json_structure_equals(self):
        pipeline_run_save_path1 = os.path.join(self.test_dir, 'pipeline_run1.yml')
        self._fit_iris_random_forest(pipeline_run_save_path=pipeline_run_save_path1)
        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path1)

        pipeline_run_save_path2 = os.path.join(self.test_dir, 'pipeline_run2.yml')
        self._fit_iris_random_forest(pipeline_run_save_path=pipeline_run_save_path2)
        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path2)

        self._assert_pipeline_runs_equal(pipeline_run_save_path1, pipeline_run_save_path2)

    def _cache_pipeline_for_rerun(self, pipeline_path, cache_dir=None):
        """make pipeline searchable by id in test_dir"""
        with open(pipeline_path, 'r') as f:
            pipeline = utils.yaml_load(f)
        if cache_dir is None:
            cache_dir = self.test_dir
        temp_pipeline_path = os.path.join(cache_dir, pipeline['id'] + '.yml')
        with open(temp_pipeline_path, 'w') as f:
            utils.yaml_dump(pipeline, f)

    @staticmethod
    def _generate_seed():
        return random.randint(2**31, 2**32-1)

    def test_fit_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_save_path = self._get_pipeline_run_save_path()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, hyperparams=hyperparams,
                random_seed=random_seed, context=metadata_base.Context.TESTING,
            )

        with open(pipeline_run_save_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)

        self._cache_pipeline_for_rerun(pipeline_path)

        pipeline_rerun_save_path = self._get_pipeline_rerun_save_path()

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit',
            '--input-run',
            pipeline_run_save_path,
            '--output-run',
            pipeline_rerun_save_path,
        ]
        self._call_cli_runtime_without_fail(rerun_arg)

        self._assert_valid_saved_pipeline_runs(pipeline_rerun_save_path)
        self._assert_pipeline_runs_equal(pipeline_run_save_path, pipeline_rerun_save_path)

    def test_produce_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_save_path = self._get_pipeline_run_save_path()
        fitted_pipeline_path = os.path.join(self.test_dir, 'iris-pipeline')

        self._fit_iris_random_forest(fitted_pipeline_path=fitted_pipeline_path)
        self.assertTrue(os.path.isfile(fitted_pipeline_path))

        arg = [
            '',
            'runtime',
            'produce',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--test-input',
            dataset_path,
            '--output-run',
            pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)
        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._cache_pipeline_for_rerun(pipeline_path)

        pipeline_rerun_save_path = self._get_pipeline_rerun_save_path()

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'produce',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--input-run',
            pipeline_run_save_path,
            '--output-run',
            pipeline_rerun_save_path,
        ]
        self._call_cli_runtime_without_fail(rerun_arg)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_save_path)

        self._assert_pipeline_runs_equal(pipeline_run_save_path, pipeline_rerun_save_path)

    def _assert_scores_equal(self, scores_path, rescores_path):
        scores = pandas.read_csv(scores_path)
        rescores = pandas.read_csv(rescores_path)
        self.assertTrue(scores.equals(rescores), '\n{}\n\n{}'.format(scores, rescores))

    def _assert_scores_equal_pipeline_run(self, scores_path, pipeline_run_save_path):
        scores = pandas.read_csv(scores_path)
        scores.drop('fold', axis=1, inplace=True, errors='ignore')
        scores_no_seed = scores.drop('randomSeed', axis=1, errors='ignore')

        with open(pipeline_run_save_path) as f:
            # TODO: always use -1?
            pipeline_run = list(utils.yaml_load_all(f))[-1]
            self.assertEqual(pipeline_run['run']['phase'], metadata_base.PipelineRunPhase.PRODUCE.name)
        # TODO: clean up preprocessing?
        pipeline_run_scores_df = pandas.DataFrame(pipeline_run['run']['results']['scores'])
        # TODO: is it possible to make pipeline run schema more compatible with scores csv schema?
        pipeline_run_scores_df['metric'] = pipeline_run_scores_df['metric'].map(lambda cell: cell['metric'])
        pipeline_run_scores_df = pipeline_run_scores_df[scores_no_seed.columns.tolist()]

        pandas.testing.assert_frame_equal(scores_no_seed, pipeline_run_scores_df)
        self.assertEqual(scores['randomSeed'].iloc[0], pipeline_run['random_seed'])

    def test_score_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_save_path = self._get_pipeline_run_save_path()
        fitted_pipeline_path = os.path.join(self.test_dir, 'iris-pipeline')
        scores_path = os.path.join(self.test_dir, 'scores.csv')

        random_seed = self._generate_seed()
        metrics = runtime.get_metrics_from_list(['ACCURACY', 'F1_MACRO'])
        scoring_params = {'add_normalized_scores': 'false'}
        scoring_random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, random_seed=random_seed,
                context=metadata_base.Context.TESTING,
            )
            with open(fitted_pipeline_path, 'wb') as f:
                pickle.dump(fitted_pipeline, f)

            predictions, produce_result = runtime.produce(fitted_pipeline, inputs)

            scores, score_result = runtime.score(
                predictions, inputs, scoring_pipeline=scoring_pipeline,
                problem_description=problem, metrics=metrics, predictions_random_seed=random_seed,
                context=metadata_base.Context.TESTING, scoring_params=scoring_params,
                random_seed=scoring_random_seed
            )

            self.assertFalse(score_result.has_error(), score_result.error)

            scores.to_csv(scores_path)

            runtime.combine_pipeline_runs(
                produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=inputs,
                metrics=metrics, scores=scores
            )
            with open(pipeline_run_save_path, 'w') as f:
                produce_result.pipeline_run.to_yaml(f)

        self.assertTrue(os.path.isfile(fitted_pipeline_path))
        self.assertTrue(os.path.isfile(scores_path), 'scores were not generated')
        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        dataframe = pandas.read_csv(scores_path)

        self.assertEqual(list(dataframe.columns), ['metric', 'value', 'randomSeed'])
        self.assertEqual(dataframe.values.tolist(), [['ACCURACY', 1.0, random_seed], ['F1_MACRO', 1.0, random_seed]])

        self._cache_pipeline_for_rerun(pipeline_path)

        pipeline_rerun_save_path = self._get_pipeline_rerun_save_path()
        rescores_path = self._get_rescores_path()

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'score',
            '--fitted-pipeline',
            fitted_pipeline_path,
            '--input-run',
            pipeline_run_save_path,
            '--output-run',
            pipeline_rerun_save_path,
            '--scores',
            rescores_path,
        ]
        self._call_cli_runtime_without_fail(rerun_arg)
        self.assertTrue(os.path.isfile(pipeline_rerun_save_path))
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_save_path)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_rerun_save_path)
        self._assert_pipeline_runs_equal(pipeline_run_save_path, pipeline_rerun_save_path)

    def test_fit_produce_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_save_path = self._get_pipeline_run_save_path()

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, hyperparams=hyperparams,
                random_seed=random_seed, context=metadata_base.Context.TESTING,
            )
            predictions, produce_result = runtime.produce(fitted_pipeline, inputs)

        with open(pipeline_run_save_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)
            produce_result.pipeline_run.to_yaml(f, appending=True)

        self._cache_pipeline_for_rerun(pipeline_path)

        pipeline_rerun_save_path = self._get_pipeline_rerun_save_path()

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            '--strict-digest',
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit-produce',
            '--input-run',
            pipeline_run_save_path,
            '--output-run',
            pipeline_rerun_save_path,
        ]
        self._call_cli_runtime_without_fail(rerun_arg)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_save_path)

        self._assert_pipeline_runs_equal(pipeline_run_save_path, pipeline_rerun_save_path)

    def test_fit_score_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        pipeline_run_save_path = self._get_pipeline_run_save_path()
        scores_path = self._get_scores_path()

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()
        metrics = runtime.get_metrics_from_list(['ACCURACY', 'F1_MACRO'])
        scoring_params = {'add_normalized_scores': 'false'}
        scoring_random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            fitted_pipeline, predictions, fit_result = runtime.fit(
                pipeline, inputs, problem_description=problem, hyperparams=hyperparams,
                random_seed=random_seed, context=metadata_base.Context.TESTING,
            )
            self.assertFalse(fit_result.has_error(), fit_result.error)

            predictions, produce_result = runtime.produce(fitted_pipeline, inputs)
            self.assertFalse(produce_result.has_error(), produce_result.error)

            scores, score_result = runtime.score(
                predictions, inputs, scoring_pipeline=scoring_pipeline,
                problem_description=problem, metrics=metrics,
                predictions_random_seed=fitted_pipeline.random_seed,
                context=metadata_base.Context.TESTING, scoring_params=scoring_params, random_seed=scoring_random_seed
            )

            self.assertFalse(score_result.has_error(), score_result.error)
            scores.to_csv(scores_path)

            runtime.combine_pipeline_runs(
                produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run, score_inputs=inputs,
                metrics=metrics, scores=scores
            )

        with open(pipeline_run_save_path, 'w') as f:
            fit_result.pipeline_run.to_yaml(f)
            produce_result.pipeline_run.to_yaml(f, appending=True)

        self._assert_valid_saved_pipeline_runs(pipeline_run_save_path)

        self._cache_pipeline_for_rerun(pipeline_path)

        pipeline_rerun_save_path = self._get_pipeline_rerun_save_path()
        rescores_path = self._get_rescores_path()

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            '--strict-digest',
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'fit-score',
            '--input-run',
            pipeline_run_save_path,
            '--scores',
            rescores_path,
            '--output-run',
            pipeline_rerun_save_path,
        ]
        self._call_cli_runtime_without_fail(rerun_arg)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_save_path)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_rerun_save_path)
        self._assert_pipeline_runs_equal(pipeline_run_save_path, pipeline_rerun_save_path)

    def test_evaluate_rerun(self):
        dataset_path = self._get_iris_dataset_path()
        problem_path = self._get_iris_problem_path()
        pipeline_path = self._get_random_forest_pipeline_path()
        data_pipeline_path = self._get_train_test_split_data_pipeline_path()
        pipeline_run_save_path = self._get_pipeline_run_save_path()
        scores_path = self._get_scores_path()

        hyperparams = [{}, {}, {'n_estimators': 19}, {}]
        random_seed = self._generate_seed()
        metrics = runtime.get_metrics_from_list(['ACCURACY', 'F1_MACRO'])
        scoring_params = {'add_normalized_scores': 'false'}
        scoring_random_seed = self._generate_seed()
        data_params = {'shuffle': 'true', 'stratified': 'true', 'train_score_ratio': '0.59'}
        data_random_seed = self._generate_seed()

        problem = problem_module.get_problem(problem_path)
        inputs = [dataset_module.get_dataset(dataset_path)]
        with open(pipeline_path) as f:
            pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(data_pipeline_path) as f:
            data_pipeline = pipeline_module.Pipeline.from_yaml(f)
        with open(runtime.DEFAULT_SCORING_PIPELINE_PATH) as f:
            scoring_pipeline = pipeline_module.Pipeline.from_yaml(f)

        with utils.silence():
            dummy_runtime_environment = pipeline_run_module.RuntimeEnvironment(worker_id='dummy worker id')

            all_scores, all_results = runtime.evaluate(
                pipeline, inputs, data_pipeline=data_pipeline, scoring_pipeline=scoring_pipeline,
                problem_description=problem, data_params=data_params, metrics=metrics,
                context=metadata_base.Context.TESTING, scoring_params=scoring_params,
                hyperparams=hyperparams, random_seed=random_seed,
                data_random_seed=data_random_seed, scoring_random_seed=scoring_random_seed,
                runtime_environment=dummy_runtime_environment,
            )

            self.assertEqual(len(all_scores), 1)
            scores = runtime.combine_folds(all_scores)
            scores.to_csv(scores_path)

            if any(result.has_error() for result in all_results):
                self.fail([result.error for result in all_results if result.has_error()][0])

        with open(pipeline_run_save_path, 'w') as f:
            for i, pipeline_run in enumerate(all_results.pipeline_runs):
                pipeline_run.to_yaml(f, appending=i>0)

        self._cache_pipeline_for_rerun(pipeline_path)
        self._cache_pipeline_for_rerun(data_pipeline_path)

        pipeline_rerun_save_path = self._get_pipeline_rerun_save_path()
        rescores_path = self._get_rescores_path()

        rerun_arg = [
            '',
            '--pipelines-path',
            self.test_dir,
            'runtime',
            '--datasets',
            TEST_DATA_DIR,
            'evaluate',
            '--input-run',
            pipeline_run_save_path,
            '--output-run',
            pipeline_rerun_save_path,
            '--scores',
            rescores_path,
        ]
        self._call_cli_runtime_without_fail(rerun_arg)
        self._assert_valid_saved_pipeline_runs(pipeline_rerun_save_path)
        self._assert_scores_equal(scores_path, rescores_path)
        self._assert_scores_equal_pipeline_run(scores_path, pipeline_rerun_save_path)
        self._assert_pipeline_runs_equal(pipeline_run_save_path, pipeline_rerun_save_path)

    # See: https://gitlab.com/datadrivendiscovery/d3m/issues/406
    # TODO: Test rerun validation code (that we throw exceptions on invalid pipeline runs).
    # TODO: Test rerun with multiple inputs (non-standard pipeline).
    # TODO: Test rerun without problem description.
    # TODO: Test evaluate rerun with data split file.

    def test_validate_gzipped_pipeline_run(self):
        # First, generate the pipeline run file
        pipeline_run_save_path = self._get_pipeline_run_save_path()
        gzip_pipeline_run_save_path = '{pipeline_run_save_path}.gz'.format(pipeline_run_save_path=pipeline_run_save_path)
        fitted_pipeline_path = os.path.join(self.test_dir, 'fitted-pipeline')
        self._fit_iris_random_forest(
            fitted_pipeline_path=fitted_pipeline_path, pipeline_run_save_path=pipeline_run_save_path
        )

        # Second, gzip the pipeline run file
        with open(pipeline_run_save_path, 'rb') as file_in:
            with gzip.open(gzip_pipeline_run_save_path, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
        os.remove(pipeline_run_save_path)

        # Third, ensure that calling 'pipeline-run validate' on the gzipped pipeline run file is successful
        arg = [
            '',
            'pipeline-run',
            'validate',
            gzip_pipeline_run_save_path,
        ]
        self._call_cli_runtime_without_fail(arg)

    def test_help_message(self):
        arg = [
            '',
            'runtime',
            'fit',
            '--version',
        ]

        with io.StringIO() as buffer:
            with contextlib.redirect_stderr(buffer):
                with self.assertRaises(SystemExit):
                    cli.main(arg)

            help = buffer.getvalue()
            self.assertTrue('usage: d3m runtime fit' in help, help)


if __name__ == '__main__':
    unittest.main()
