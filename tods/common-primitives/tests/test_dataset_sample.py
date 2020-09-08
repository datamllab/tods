import os
import pickle
import unittest
import pandas as pd

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_sample


class DatasetSamplePrimitiveTestCase(unittest.TestCase):
    def test_produce(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_sample.DatasetSamplePrimitive.metadata.get_hyperparams()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

        sample_sizes = [0.1, 0.5, 0.9, 4, 22, 40]
        dataset_sizes = [4, 22, 40, 4, 22, 40]
        for s, d in zip(sample_sizes, dataset_sizes):
            primitive = dataset_sample.DatasetSamplePrimitive(hyperparams=hyperparams_class.defaults().replace({
                'sample_size': s,
            }))
            result = primitive.produce(inputs=dataset).value
            self.assertEqual(len(result['learningData'].iloc[:, 0]), d, s)

    def test_empty_test_set(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # set target columns to '' to imitate test dataset
        dataset['learningData']['species'] = ''

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

        hyperparams_class = dataset_sample.DatasetSamplePrimitive.metadata.get_hyperparams()

        # check that no rows are sampled
        sample_sizes = [0.1, 0.5, 0.9]
        for s in sample_sizes:
            primitive = dataset_sample.DatasetSamplePrimitive(hyperparams=hyperparams_class.defaults().replace({
                'sample_size': s,
            }))
            result = primitive.produce(inputs=dataset).value
            self.assertEqual(len(result['learningData'].iloc[:, 0]), 150, s)


if __name__ == '__main__':
    unittest.main()
