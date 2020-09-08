import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import no_split


class NoSplitDatasetSplitPrimitiveTestCase(unittest.TestCase):
    def test_produce_train(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = no_split.NoSplitDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = no_split.NoSplitDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults())

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 150)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [str(i) for i in range(150)])

    def test_produce_score(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = no_split.NoSplitDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = no_split.NoSplitDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults())

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 150)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [str(i) for i in range(150)])


if __name__ == '__main__':
    unittest.main()
