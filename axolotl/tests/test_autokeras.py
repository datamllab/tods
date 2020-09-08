import pathlib
import shutil
import sys
import unittest

import os
import tempfile

from axolotl.algorithms.autokeras_search import AutoKerasSearch
from axolotl.backend.simple import SimpleRunner

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from d3m.metadata import problem as problem_module
from d3m import container as container_module


class TestAutoKeras(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.backend = SimpleRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir)


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fit(self):
        test_data = os.path.join(PROJECT_ROOT, 'tests', 'data')
        dataset_name = 'image_dataset_2'

        dataset_path = os.path.join(
            test_data, 'datasets', dataset_name, 'datasetDoc.json')
        dataset = self.__get_dataset(dataset_path)

        problem_path = os.path.join(
            test_data, 'problems', dataset_name.replace('dataset', 'problem'), 'problemDoc.json')
        problem = self.__get_problem(problem_path)

        tuner_base = AutoKerasSearch(problem, backend=self.backend, max_trials=1, directory=self.test_dir)
        pipeline_result = tuner_base.search_fit(input_data=[dataset], time_limit=1000)
        # TODO https://gitlab.com/datadrivendiscovery/jpl-primitives/-/issues/41
        self.assertNotEqual(pipeline_result.error, None)

    def _fit_cifar10(self):
        test_data = os.path.join('/data/d3m/datasets/seed_datasets_current')
        dataset_name = '124_174_cifar10_MIN_METADATA'

        dataset_path = os.path.join(
            test_data, dataset_name, '{}_dataset'.format(dataset_name), 'datasetDoc.json')
        dataset = self.__get_dataset(dataset_path)

        problem_path = os.path.join(
            test_data, dataset_name, '{}_problem'.format(dataset_name), 'problemDoc.json')
        problem = self.__get_problem(problem_path)

        tuner_base = AutoKerasSearch(problem, backend=self.backend, max_trials=1, directory=self.test_dir)
        pipeline_result = tuner_base.search_fit(input_data=[dataset], time_limit=1000)
        # TODO https://gitlab.com/datadrivendiscovery/jpl-primitives/-/issues/41
        self.assertNotEqual(pipeline_result.error, None)

    def __get_uri(self, path):
        return pathlib.Path(os.path.abspath(path)).as_uri()

    def __get_problem(self, problem_path):
        problem_uri = self.__get_uri(problem_path)
        problem = problem_module.Problem.load(problem_uri)
        return problem

    def __get_dataset(self, dataset_path):
        dataset_uri = self.__get_uri(dataset_path)
        dataset = container_module.dataset.get_dataset(dataset_uri)
        return dataset


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for test_case in (
        'test_fit',
    ):
        suite.addTest(TestAutoKeras(test_case))
    unittest.TextTestRunner(verbosity=2).run(suite)
