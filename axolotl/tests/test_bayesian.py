import pathlib

import ray
import shutil
import sys
import unittest

import os
import tempfile
from axolotl.backend.ray import RayRunner

from axolotl.algorithms.bayesian_search import BayesianSearch
from axolotl.backend.simple import SimpleRunner

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from d3m.metadata import problem as problem_module
from d3m import container as container_module
from axolotl.utils import pipeline as pipeline_utils


class TestBayesianSearch(unittest.TestCase):
    def setUp(self):
        self.test_data = os.path.join(PROJECT_ROOT, 'tests', 'data')
        dataset_name = 'iris_dataset_1'
        problem = self.__get_problem(dataset_name)
        self.problem = problem
        self.dataset = self.__get_dataset(dataset_name)
        self.test_dir = tempfile.mkdtemp()
        backend = SimpleRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir)
        self.tuner_base = BayesianSearch(problem, backend=backend, max_trials=10, directory=self.test_dir,
                                         num_initial_points=5)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fit(self):
        _, pipeline_result = self.tuner_base.search_fit(input_data=[self.dataset], time_limit=60)
        self.assertEqual(pipeline_result.error, None)

    def test_fit_svc(self):
        pipeline_info = os.path.join(os.path.dirname(__file__),  'resources', 'svc_pipeline.json')
        pipeline = pipeline_utils.load_pipeline(pipeline_info)
        _, pipeline_result = self.tuner_base.search_fit(input_data=[self.dataset], time_limit=60,
                                                        pipeline_candidates=[pipeline])
        self.assertEqual(pipeline_result.error, None)

    def test_fit_lr(self):
        pipeline_info = os.path.join(os.path.dirname(__file__),  'resources', 'logistic_regeression.json')
        pipeline = pipeline_utils.load_pipeline(pipeline_info)
        _, pipeline_result = self.tuner_base.search_fit(input_data=[self.dataset], time_limit=60,
                                                        pipeline_candidates=[pipeline])
        self.assertEqual(pipeline_result.error, None)

    def test_fit_ray(self):
        if not ray.is_initialized():
            ray.init()
        backend = RayRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir)
        tuner_base = BayesianSearch(self.problem, backend=backend, max_trials=30, directory=self.test_dir,
                                    num_initial_points=5)
        _, pipeline_result = tuner_base.search_fit(input_data=[self.dataset], time_limit=100)
        self.assertEqual(pipeline_result.error, None)
        ray.shutdown()

    def __get_uri(self, path):
        return pathlib.Path(os.path.abspath(path)).as_uri()

    def __get_problem(self, dataset_name):
        problem_path = os.path.join(
            self.test_data, 'problems', dataset_name.replace('dataset', 'problem'), 'problemDoc.json')
        problem_uri = self.__get_uri(problem_path)
        problem = problem_module.Problem.load(problem_uri)
        return problem

    def __get_dataset(self, dataset_name):
        dataset_path = os.path.join(
            self.test_data, 'datasets', dataset_name, 'datasetDoc.json')
        dataset_uri = self.__get_uri(dataset_path)
        dataset = container_module.dataset.get_dataset(dataset_uri)
        return dataset


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for test_case in (
        'test_fit',
        'test_fit_ray',
        'test_fit_lr',
        'test_fit_svc',
    ):
        suite.addTest(TestBayesianSearch(test_case))
    unittest.TextTestRunner(verbosity=2).run(suite)
