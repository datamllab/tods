import os
import unittest

import numpy

from d3m import container, exceptions
from d3m.metadata import base as metadata_base
from d3m.contrib.primitives import compute_scores


class ComputeScoresPrimitiveTestCase(unittest.TestCase):
    def test_regression(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        random = numpy.random.RandomState(42)

        # Create a synthetic prediction DataFrame.
        d3mIndex = dataset['learningData'].iloc[:, 0].astype(int)
        value = random.randn(len(d3mIndex))
        predictions = container.DataFrame({'d3mIndex': d3mIndex, 'value': value}, generate_metadata=True)
        shuffled_predictions = predictions.reindex(random.permutation(predictions.index)).reset_index(drop=True)

        hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.get_hyperparams()
        metrics_class = hyperparams_class.configuration['metrics'].elements
        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'MEAN_SQUARED_ERROR',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'ROOT_MEAN_SQUARED_ERROR',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'MEAN_ABSOLUTE_ERROR',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'R_SQUARED',
                'pos_label': None,
                'k': None,
            })],
        }))

        for name, pred in zip(['predictions', 'shuffled_predictions'], [predictions, shuffled_predictions]):
            scores = primitive.produce(inputs=pred, score_dataset=dataset).value
            self.assertEqual(scores.values.tolist(), [
                ['MEAN_SQUARED_ERROR', 3112.184932446708, 0.08521485450672399],
                ['ROOT_MEAN_SQUARED_ERROR', 55.786960236660214, 0.9721137517700256],
                ['MEAN_ABSOLUTE_ERROR', 54.579668078204385, 0.9727169385086356],
                ['R_SQUARED', -22.62418041588221, 0.9881884591239001],
            ], name)

            self.assertEqual(scores.metadata.query_column(0)['name'], 'metric', name)
            self.assertEqual(scores.metadata.query_column(1)['name'], 'value', name)

    def test_multivariate(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'multivariate_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        random = numpy.random.RandomState(42)

        # Create a synthetic prediction DataFrame.
        d3mIndex = dataset['learningData'].iloc[:, 0].astype(int)
        amplitude = random.randn(len(d3mIndex))
        lengthscale = random.randn(len(d3mIndex))
        predictions = container.DataFrame({'d3mIndex': d3mIndex, 'amplitude': amplitude, 'lengthscale': lengthscale}, generate_metadata=True)
        shuffled_predictions = predictions.reindex(random.permutation(predictions.index)).reset_index(drop=True)

        hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.get_hyperparams()
        metrics_class = hyperparams_class.configuration['metrics'].elements
        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'MEAN_SQUARED_ERROR',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'ROOT_MEAN_SQUARED_ERROR',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'MEAN_ABSOLUTE_ERROR',
                'pos_label': None,
                'k': None,
            })],
        }))

        for name, pred in zip(['predictions', 'shuffled_predictions'], [predictions, shuffled_predictions]):
            scores = primitive.produce(inputs=pred, score_dataset=dataset).value
            self.assertEqual(scores.values.tolist(), [
                ['MEAN_SQUARED_ERROR', 1.7627871219522482, 0.9991186066672619],
                ['ROOT_MEAN_SQUARED_ERROR', 1.3243591896125282, 0.9993378205019783],
                ['MEAN_ABSOLUTE_ERROR', 1.043095768817859, 0.9994784521628801],
            ], name)

            self.assertEqual(scores.metadata.query_column(0)['name'], 'metric', name)
            self.assertEqual(scores.metadata.query_column(1)['name'], 'value', name)

    def test_classification(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        random = numpy.random.RandomState(42)

        # Create a synthetic prediction DataFrame.
        d3mIndex = dataset['learningData'].iloc[:, 0].astype(int)
        species = random.choice(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], len(d3mIndex))
        predictions = container.DataFrame({'d3mIndex': d3mIndex, 'species': species}, generate_metadata=True)
        shuffled_predictions = predictions.reindex(random.permutation(predictions.index)).reset_index(drop=True)

        hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.get_hyperparams()
        metrics_class = hyperparams_class.configuration['metrics'].elements
        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'ACCURACY',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'F1_MICRO',
                'pos_label': None,
                'k': None,
            }), metrics_class({
                'metric': 'F1_MACRO',
                'pos_label': None,
                'k': None,
            })],
        }))

        for name, pred in zip(['predictions', 'shuffled_predictions'], [predictions, shuffled_predictions]):
            scores = primitive.produce(inputs=pred, score_dataset=dataset).value
            self.assertEqual(scores.values.tolist(), [
                ['ACCURACY', 0.4066666666666667, 0.4066666666666667],
                ['F1_MICRO', 0.4066666666666667, 0.4066666666666667],
                ['F1_MACRO', 0.4051068540623797, 0.4051068540623797],
            ], name)

            self.assertEqual(scores.metadata.query_column(0)['name'], 'metric', name)
            self.assertEqual(scores.metadata.query_column(1)['name'], 'value', name)

    # TODO: Test also when there is both "color_not_class" and "bounding_polygon_area" targets predicted.
    def test_object_detection_just_bounding_polygon(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'object_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        random = numpy.random.RandomState(42)

        # Create a synthetic prediction DataFrame.
        predictions = container.DataFrame([
            [0, '330,463,330,505,387,505,387,463', 0.0739],
            [0, '420,433,420,498,451,498,451,433', 0.091],
            [0, '328,465,328,540,403,540,403,465', 0.1008],
            [0, '480,477,480,522,508,522,508,477', 0.1012],
            [0, '357,460,357,537,417,537,417,460', 0.1058],
            [0, '356,456,356,521,391,521,391,456', 0.0843],
            [1, '345,460,345,547,415,547,415,460', 0.0539],
            [1, '381,362,381,513,455,513,455,362', 0.0542],
            [1, '382,366,382,422,416,422,416,366', 0.0559],
            [1, '730,463,730,583,763,583,763,463', 0.0588],
        ], columns=['d3mIndex', 'bounding_polygon_area', 'confidence'], generate_metadata=True)
        shuffled_predictions = predictions.reindex(random.permutation(predictions.index)).reset_index(drop=True)

        hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.get_hyperparams()
        metrics_class = hyperparams_class.configuration['metrics'].elements
        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'OBJECT_DETECTION_AVERAGE_PRECISION',
                'pos_label': None,
                'k': None,
            })],
        }))

        for name, pred in zip(['predictions', 'shuffled_predictions'], [predictions, shuffled_predictions]):
            scores = primitive.produce(inputs=pred, score_dataset=dataset).value
            self.assertEqual(scores.values.tolist(), [
                ['OBJECT_DETECTION_AVERAGE_PRECISION', 0.125, 0.125],
            ], name)

            self.assertEqual(scores.metadata.query_column(0)['name'], 'metric')
            self.assertEqual(scores.metadata.query_column(1)['name'], 'value')
            self.assertEqual(scores.metadata.query_column(2)['name'], 'normalized')

    def test_all_labels(self):
        truth = container.DataFrame([
            [3, 'happy-pleased'],
            [3, 'relaxing-calm'],
            [7, 'amazed-suprised'],
            [7, 'happy-pleased'],
            [13, 'quiet-still'],
            [13, 'sad-lonely'],
        ], columns=['d3mIndex', 'class_label'])

        truth_dataset = container.Dataset({'learningData': truth}, generate_metadata=True)

        truth_dataset.metadata = truth_dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        truth_dataset.metadata = truth_dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Target')
        truth_dataset.metadata = truth_dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

        predictions = container.DataFrame([
            [3, 'happy-pleased'],
            [3, 'sad-lonely'],
            [7, 'amazed-suprised'],
            [7, 'happy-pleased'],
            [13, 'quiet-still'],
            [13, 'happy-pleased'],
        ], columns=['d3mIndex', 'class_label'], generate_metadata=True)

        hyperparams_class = compute_scores.ComputeScoresPrimitive.metadata.get_hyperparams()
        metrics_class = hyperparams_class.configuration['metrics'].elements
        all_labels_class = hyperparams_class.configuration['all_labels'].elements
        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'HAMMING_LOSS',
                'pos_label': None,
                'k': None,
            })],
        }))

        scores = primitive.produce(inputs=predictions, score_dataset=truth_dataset).value
        self.assertEqual(scores.values.tolist(), [
            ['HAMMING_LOSS', 0.26666666666666666, 0.7333333333333334],
        ])

        self.assertEqual(scores.metadata.query_column(0)['name'], 'metric')
        self.assertEqual(scores.metadata.query_column(1)['name'], 'value')
        self.assertEqual(scores.metadata.query_column(2)['name'], 'normalized')

        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'HAMMING_LOSS',
                'pos_label': None,
                'k': None,
            })],
            'all_labels': [all_labels_class({
                'column_name': 'class_label',
                'labels': ['happy-pleased', 'relaxing-calm', 'amazed-suprised', 'quiet-still', 'sad-lonely', 'foobar'],
            })],
        }))

        scores = primitive.produce(inputs=predictions, score_dataset=truth_dataset).value
        self.assertEqual(scores.values.tolist(), [
            ['HAMMING_LOSS', 0.2222222222222222, 0.7777777777777778],
        ])

        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'HAMMING_LOSS',
                'pos_label': None,
                'k': None,
            })],
            'all_labels': [all_labels_class({
                'column_name': 'class_label',
                'labels': ['happy-pleased', 'relaxing-calm', 'amazed-suprised'],
            })],
        }))

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'Truth contains extra labels'):
            primitive.produce(inputs=predictions, score_dataset=truth_dataset)

        truth_dataset.metadata = truth_dataset.metadata.update_column(1, {
            'all_distinct_values': ['happy-pleased', 'relaxing-calm', 'amazed-suprised', 'quiet-still', 'sad-lonely', 'foobar'],
        }, at=('learningData',))

        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'HAMMING_LOSS',
                'pos_label': None,
                'k': None,
            })],
        }))

        scores = primitive.produce(inputs=predictions, score_dataset=truth_dataset).value
        self.assertEqual(scores.values.tolist(), [
            ['HAMMING_LOSS', 0.2222222222222222, 0.7777777777777778],
        ])

        truth_dataset.metadata = truth_dataset.metadata.update_column(1, {
            'all_distinct_values': ['happy-pleased', 'relaxing-calm', 'amazed-suprised'],
        }, at=('learningData',))

        primitive = compute_scores.ComputeScoresPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'metrics': [metrics_class({
                'metric': 'HAMMING_LOSS',
                'pos_label': None,
                'k': None,
            })],
        }))

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'Truth contains extra labels'):
            primitive.produce(inputs=predictions, score_dataset=truth_dataset)


if __name__ == '__main__':
    unittest.main()
