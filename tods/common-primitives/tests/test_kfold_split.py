import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import kfold_split


class KFoldDatasetSplitPrimitiveTestCase(unittest.TestCase):
    def test_produce_train(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = kfold_split.KFoldDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = kfold_split.KFoldDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': 10,
            'shuffle': True,
            'delete_recursive': True,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
            self.assertEqual(len(dataset), 4)

        self.assertEqual(results[0]['codes'].shape[0], 3)
        self.assertEqual(results[1]['codes'].shape[0], 3)

        self.assertEqual(set(results[0]['codes'].iloc[:, 0]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 40)
        self.assertEqual(set(results[0]['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 2]), {'aaa', 'bbb', 'ccc', 'ddd', 'eee'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'1990', '2000', '2010'})

        self.assertEqual(set(results[1]['codes'].iloc[:, 0]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(len(results[1]['learningData'].iloc[:, 0]), 40)
        self.assertEqual(set(results[1]['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 2]), {'aaa', 'bbb', 'ccc', 'ddd', 'eee'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'1990', '2000', '2010'})

    def test_produce_score(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = kfold_split.KFoldDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = kfold_split.KFoldDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': 10,
            'shuffle': True,
            'delete_recursive': True,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
            self.assertEqual(len(dataset), 4)

        self.assertEqual(set(results[0]['codes'].iloc[:, 0]), {'AAA', 'BBB'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 0]), {'5', '11', '28', '31', '38'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 1]), {'AAA', 'BBB'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 2]), {'aaa', 'bbb', 'ddd', 'eee'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'1990', '2000'})

        self.assertEqual(set(results[1]['codes'].iloc[:, 0]), {'BBB', 'CCC'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 0]), {'12', '26', '29', '32', '39'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 1]), {'BBB', 'CCC'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 2]), {'bbb', 'ccc', 'ddd', 'eee'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'1990', '2000', '2010'})


if __name__ == '__main__':
    unittest.main()
