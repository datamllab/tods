import os
import pathlib
import unittest
import sys
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

from d3m.runtime import Runtime
from d3m.metadata import base as metadata_base, problem as problem_module
from d3m import container as container_module

import axolotl.predefined_pipelines as predefined_pipelines


class TestPredefined(unittest.TestCase):
    def setUp(self):
        self.test_data = os.path.join(PROJECT_ROOT, 'tests', 'data')

    def tearDown(self):
        pass

    def test_fetch_from_file(self):
        dataset_name = 'iris_dataset_1'
        problem = self.__get_problem(dataset_name)
        dataset = self.__get_dataset(dataset_name)
        predefined_path = os.path.join(PROJECT_ROOT, 'axolotl/utils/resources/default_pipelines.json')
        pipelines = predefined_pipelines.fetch_from_file(problem, predefined_path)
        self.assertNotEqual(len(pipelines), 0)
        result = self.__run_pipeline(pipelines[0], dataset)
        result.check_success()
        self.assertEqual(result.error, None)

    def test__fetch_from_preprocessors(self):
        dataset_name = 'iris_dataset_1'
        problem = self.__get_problem(dataset_name)
        dataset = self.__get_dataset(dataset_name)
        pipelines = predefined_pipelines._fetch_from_preprocessors(dataset, problem)
        self.assertNotEqual(len(pipelines), 0)
        result = self.__run_pipeline(pipelines[0], dataset)
        result.check_success()
        self.assertEqual(result.error, None)

    def test_fetch(self):
        dataset_name = 'iris_dataset_1'
        problem = self.__get_problem(dataset_name)
        dataset = self.__get_dataset(dataset_name)
        pipelines = predefined_pipelines.fetch(dataset, problem)
        self.assertNotEqual(len(pipelines), 0)
        result = self.__run_pipeline(pipelines[-1], dataset)
        result.check_success()
        self.assertEqual(result.error, None)

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

    def __run_pipeline(self, pipeline_description, data, volume_dir='/volumes'):
        runtime = Runtime(pipeline=pipeline_description, context=metadata_base.Context.TESTING, volumes_dir=volume_dir)
        fit_result = runtime.fit([data])
        return fit_result


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for test_case in (
        'test_fetch_from_file',
        'test__fetch_from_preprocessors',
        'test_fetch',

    ):
        suite.addTest(TestPredefined(test_case))
    unittest.TextTestRunner(verbosity=2).run(suite)
