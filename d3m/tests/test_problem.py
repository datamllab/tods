import os.path
import pickle
import unittest

from d3m import utils
from d3m.metadata import problem, pipeline_run


class TestProblem(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None

        problem_doc_path = os.path.join(os.path.dirname(__file__), 'data', 'problems', 'iris_problem_1', 'problemDoc.json')

        problem_uri = 'file://{problem_doc_path}'.format(problem_doc_path=problem_doc_path)

        problem_description = problem.Problem.load(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), {
            'id': 'iris_problem_1',
            'digest': '1a12135422967aa0de0c4629f4f58d08d39e97f9133f7b50da71420781aa18a5',
            'version': '4.0.0',
            'location_uris': [
                problem_uri,
            ],
            'name': 'Distinguish Iris flowers',
            'description': 'Distinguish Iris flowers of three related species.',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': [problem.TaskKeyword.CLASSIFICATION, problem.TaskKeyword.MULTICLASS],
                'performance_metrics': [
                    {
                        'metric': problem.PerformanceMetric.ACCURACY,
                    }
                ]
            },
            'inputs': [
                {
                    'dataset_id': 'iris_dataset_1',
                    'targets': [
                        {
                            'target_index': 0,
                            'resource_id': 'learningData',
                            'column_index': 5,
                            'column_name': 'species',
                        }
                    ]
                }
            ],
        })

        self.assertEqual(problem_description.to_json_structure(), {
            'id': 'iris_problem_1',
            'digest': '1a12135422967aa0de0c4629f4f58d08d39e97f9133f7b50da71420781aa18a5',
            'version': '4.0.0',
            'location_uris': [
                problem_uri,
            ],
            'name': 'Distinguish Iris flowers',
            'description': 'Distinguish Iris flowers of three related species.',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': [problem.TaskKeyword.CLASSIFICATION, problem.TaskKeyword.MULTICLASS],
                'performance_metrics': [
                    {
                        'metric': problem.PerformanceMetric.ACCURACY,
                    }
                ]
            },
            'inputs': [
                {
                    'dataset_id': 'iris_dataset_1',
                    'targets': [
                        {
                            'target_index': 0,
                            'resource_id': 'learningData',
                            'column_index': 5,
                            'column_name': 'species',
                        }
                    ]
                }
            ],
        })

        self.assertEqual(problem_description.to_json_structure(), {
            'id': 'iris_problem_1',
            'digest': '1a12135422967aa0de0c4629f4f58d08d39e97f9133f7b50da71420781aa18a5',
            'version': '4.0.0',
            'location_uris': [
                problem_uri,
            ],
            'name': 'Distinguish Iris flowers',
            'description': 'Distinguish Iris flowers of three related species.',
            'schema': problem.PROBLEM_SCHEMA_VERSION,
            'problem': {
                'task_keywords': ['CLASSIFICATION', 'MULTICLASS'],
                'performance_metrics': [
                    {
                        'metric': 'ACCURACY',
                    }
                ]
            },
            'inputs': [
                {
                    'dataset_id': 'iris_dataset_1',
                    'targets': [
                        {
                            'target_index': 0,
                            'resource_id': 'learningData',
                            'column_index': 5,
                            'column_name': 'species',
                        }
                    ]
                }
            ],
        })

        pipeline_run.validate_problem(problem_description.to_json_structure(canonical=True))
        problem.PROBLEM_SCHEMA_VALIDATOR.validate(problem_description.to_json_structure(canonical=True))

    def test_conversion(self):
        problem_doc_path = os.path.join(os.path.dirname(__file__), 'data', 'problems', 'iris_problem_1', 'problemDoc.json')

        problem_uri = 'file://{problem_doc_path}'.format(problem_doc_path=problem_doc_path)

        problem_description = problem.Problem.load(problem_uri)

        self.assertEqual(problem_description.to_simple_structure(), problem.Problem.from_json_structure(problem_description.to_json_structure(), strict_digest=True).to_simple_structure())

        # Legacy.
        self.assertEqual(utils.to_json_structure(problem_description.to_simple_structure()), problem.Problem.from_json_structure(utils.to_json_structure(problem_description.to_simple_structure()), strict_digest=True).to_simple_structure())

        self.assertIs(problem.Problem.from_json_structure(problem_description.to_json_structure(), strict_digest=True)['problem']['task_keywords'][0], problem.TaskKeyword.CLASSIFICATION)

    def test_unparse(self):
        self.assertEqual(problem.TaskKeyword.CLASSIFICATION.unparse(), 'classification')
        self.assertEqual(problem.TaskKeyword.MULTICLASS.unparse(), 'multiClass')
        self.assertEqual(problem.PerformanceMetric.ACCURACY.unparse(), 'accuracy')

    def test_normalize(self):
        self.assertEqual(problem.PerformanceMetric._normalize(0, 1, 0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(0, 2, 0.5), 0.25)
        self.assertEqual(problem.PerformanceMetric._normalize(1, 2, 1.5), 0.5)

        self.assertEqual(problem.PerformanceMetric._normalize(-1, 0, -0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(-2, 0, -1.5), 0.25)
        self.assertEqual(problem.PerformanceMetric._normalize(-2, -1, -1.5), 0.5)

        self.assertEqual(problem.PerformanceMetric._normalize(1, 0, 0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(2, 0, 0.5), 0.75)
        self.assertEqual(problem.PerformanceMetric._normalize(2, 1, 1.5), 0.5)

        self.assertEqual(problem.PerformanceMetric._normalize(0, -1, -0.5), 0.5)
        self.assertEqual(problem.PerformanceMetric._normalize(0, -2, -1.5), 0.75)
        self.assertEqual(problem.PerformanceMetric._normalize(-1, -2, -1.5), 0.5)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 0.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 1000.0), 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 0, 5000.0), 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 1.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 1000.0), 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), 1, 5000.0), 0.013399004523107192)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, -1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, -0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, 1000.0), 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, 5000.0), 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('inf'), -1, 0.0), 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, 0.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, -0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, -1000.0), 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 0, -5000.0), 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, 1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, 0.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, -1000.0), 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, -5000.0), 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), 1, 0.0), 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -1.0), 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -1.5), 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -1000.0), 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(float('-inf'), -1, -5000.0), 0.013399004523107192)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 0.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 1000.0), 1 - 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('inf'), 5000.0), 1 - 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 1.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 1000.0), 1 - 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('inf'), 5000.0), 1 - 0.013399004523107192)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), -1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), -0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), 1000.0), 1 - 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), 5000.0), 1 - 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('inf'), 0.0), 1 - 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), 0.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), -0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), -1000.0), 1 - 0.5378828427399902)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(0, float('-inf'), -5000.0), 1 - 0.013385701848569713)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), 1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), 0.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), -1000.0), 1 - 0.5374897097430198)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), -5000.0), 1 - 0.01337241229216877)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(1, float('-inf'), 0.0), 1 - 0.9995000000416667)

        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -1.0), 1 - 1.0)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -1.5), 1 - 0.9997500000052083)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -1000.0), 1 - 0.5382761574524354)
        self.assertAlmostEqual(problem.PerformanceMetric._normalize(-1, float('-inf'), -5000.0), 1 - 0.013399004523107192)

    def test_pickle(self):
        value = problem.PerformanceMetric.ACCURACY

        pickled = pickle.dumps(value)
        unpickled = pickle.loads(pickled)

        self.assertEqual(value, unpickled)
        self.assertIs(value.get_class(), unpickled.get_class())


if __name__ == '__main__':
    unittest.main()
