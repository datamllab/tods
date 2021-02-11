import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from tods.common import FixedSplit


class FixedSplitDatasetSplitPrimitiveTestCase(unittest.TestCase):

    def _get_yahoo_dataset(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        return dataset

    def test_produce_train_values(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = FixedSplit.FixedSplitDatasetSplitPrimitive.metadata.get_hyperparams()

        hyperparams = hyperparams_class.defaults().replace({
            'primary_index_values': ['9', '11', '13'],
        })

        # We want to make sure "primary_index_values" is encoded just as a list and not
        # a pickle because runtime populates this primitive as a list from a split file.
        self.assertEqual(hyperparams.values_to_json_structure(), {'primary_index_values': ['9', '11', '13'], 'row_indices': [], 'delete_recursive': False})

        primitive = FixedSplit.FixedSplitDatasetSplitPrimitive(hyperparams=hyperparams)

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 1257)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [str(i) for i in range(1260) if i not in [9, 11, 13]])

    def test_produce_score_values(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = FixedSplit.FixedSplitDatasetSplitPrimitive.metadata.get_hyperparams()

        hyperparams = hyperparams_class.defaults().replace({
            'primary_index_values': ['9', '11', '13'],
        })

        # We want to make sure "primary_index_values" is encoded just as a list and not
        # a pickle because runtime populates this primitive as a list from a split file.
        self.assertEqual(hyperparams.values_to_json_structure(), {'primary_index_values': ['9', '11', '13'], 'row_indices': [], 'delete_recursive': False})

        primitive = FixedSplit.FixedSplitDatasetSplitPrimitive(hyperparams=hyperparams)

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 3)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [str(i) for i in range(150) if i in [9, 11, 13]])

    def test_produce_train_indices(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        hyperparams_class = FixedSplit.FixedSplitDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = FixedSplit.FixedSplitDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'row_indices': [9, 11, 13],
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 1257)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [str(i) for i in range(1260) if i not in [9, 11, 13]])

    def test_produce_score_indices(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = FixedSplit.FixedSplitDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = FixedSplit.FixedSplitDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'row_indices': [9, 11, 13],
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 3)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [str(i) for i in range(150) if i in [9, 11, 13]])


if __name__ == '__main__':
    unittest.main()
