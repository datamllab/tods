import ray
import json
from pathlib import Path
import unittest
import tempfile
import shutil

from d3m.metadata import problem as problem_module
from d3m import container

from axolotl.backend.ray import RayRunner
from axolotl.utils import schemas as schemas_utils, pipeline as pipeline_utils


class SimpleRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fit_produce_pipelines(self):
        pipeline = get_classification_pipeline()
        problem_description, dataset = get_data()
        ray_runner = RayRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir, n_workers=1)
        result = ray_runner.fit_pipeline(problem_description=problem_description,
                                         pipeline=pipeline, input_data=[dataset])

        self.assertEqual(result.status, 'COMPLETED')

        result = ray_runner.produce_pipeline(fitted_pipeline_id=result.fitted_pipeline_id, input_data=[dataset])
        self.assertEqual(result.status, 'COMPLETED')

    def test_evaluate_pipeline(self):
        pipeline = get_classification_pipeline()
        ray_runner = RayRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir, n_workers=1)
        problem_description, dataset = get_data()
        data_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
        scoring_pipeline = schemas_utils.get_scoring_pipeline()

        no_split = schemas_utils.DATA_PREPARATION_PARAMS['no_split']

        result = ray_runner.evaluate_pipeline(
                problem_description=problem_description, pipeline=pipeline,
                input_data=[dataset], metrics=schemas_utils.MULTICLASS_CLASSIFICATION_METRICS,
                data_preparation_pipeline=data_pipeline, scoring_pipeline=scoring_pipeline,
                data_preparation_params=no_split
        )

        self.assertEqual(result.error, None)
        self.assertEqual(result.scores.values.tolist(), [
            ['ACCURACY', 0.9133333333333333, 0.9133333333333333, 42, 0],
            ['F1_MICRO', 0.9133333333333333, 0.9133333333333333, 42, 0],
            ['F1_MACRO', 0.9123688388315397, 0.9123688388315397, 42, 0]]
        )

    def test_evaluate_pipelines(self):
        pipeline = get_classification_pipeline()
        ray_runner = RayRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir, n_workers=1)
        problem_description, dataset = get_data()
        data_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
        scoring_pipeline = schemas_utils.get_scoring_pipeline()

        no_split = schemas_utils.DATA_PREPARATION_PARAMS['no_split']

        results = ray_runner.evaluate_pipelines(
                problem_description=problem_description, pipelines=[pipeline] * 3,
                input_data=[dataset], metrics=schemas_utils.MULTICLASS_CLASSIFICATION_METRICS,
                data_preparation_pipeline=data_pipeline, scoring_pipeline=scoring_pipeline,
                data_preparation_params=no_split
        )

        for result in results:
            self.assertEqual(result.error, None)
            self.assertEqual(result.status, 'COMPLETED')


def get_classification_pipeline():
    with open(schemas_utils.PIPELINES_DB_DIR) as file:
        default_pipelines = json.load(file)

    return pipeline_utils.load_pipeline(default_pipelines['CLASSIFICATION'][0])


def get_data(dataset_name='iris_dataset_1', problem_name='iris_problem_1'):
    if problem_name:
        problem_doc_path = Path(
            Path(__file__).parent.absolute(), 'data', 'problems', problem_name, 'problemDoc.json'
        ).as_uri()
        problem_description = problem_module.get_problem(problem_doc_path)
    else:
        problem_description = None

    dataset_doc_path = Path(Path(__file__).parent.absolute(), 'data', 'datasets',
                            dataset_name, 'datasetDoc.json').as_uri()
    iris_dataset = container.dataset.get_dataset(dataset_doc_path)
    return problem_description, iris_dataset


if __name__ == '__main__':
    ray.init()
    unittest.main()
    ray.shutdown()


