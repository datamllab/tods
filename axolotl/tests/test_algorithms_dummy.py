from pathlib import Path
import unittest
import tempfile
import shutil

from d3m.metadata import problem as problem_module
from d3m import container

from axolotl.backend.simple import SimpleRunner
from axolotl.algorithms.dummy import DummySearch


class SimpleSearch(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_search_fit_produce(self):
        problem_description, dataset = get_data()

        backend = SimpleRunner(random_seed=42, volumes_dir=None, scratch_dir=self.test_dir)
        dummy_search = DummySearch(problem_description=problem_description, backend=backend)

        # check if we were able to find and fit
        fitted_pipeline, pipeline_result = dummy_search.search_fit(input_data=[dataset], time_limit=100)
        self.assertEqual(pipeline_result.error, None)

        # check first history entry
        self.assertEqual(dummy_search.history[0].scores.values.tolist()[0], [
            'ACCURACY', 0.9133333333333333, 0.9133333333333333, 42, 0])

        # test if we can produce the same training input
        pipeline_result = dummy_search.produce(fitted_pipeline, [dataset])
        self.assertEqual(pipeline_result.error, None)


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
    unittest.main()
