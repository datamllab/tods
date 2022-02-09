import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from tods.common import TrainScoreSplit 


class TrainScoreDatasetSplitPrimitiveTestCase(unittest.TestCase):
    def _get_yahoo_dataset(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        return dataset

    def test_produce_train(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 6), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = TrainScoreSplit.TrainScoreDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = TrainScoreSplit.TrainScoreDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'shuffle': True,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 945)

        column_names = ['d3mIndex', 'timestamp', 'value_0', 'value_1', 'value_2', 'value_3', 'value_4','ground_truth']
        for i in range(8):
            self.assertEqual(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, i))['name'],
                             column_names[i])

        self.assertEqual(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, 0))['semantic_types'], (
            'http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/Index'
        ))
        for i in range(2, 6):
            self.assertEqual(
                results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, i))['semantic_types'], ('http://schema.org/Float',)
            )
        self.assertEqual(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, 7))['semantic_types'],(
            'http://schema.org/Integer',
            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget',
        ))

    def test_produce_score(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 6), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = TrainScoreSplit.TrainScoreDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = TrainScoreSplit.TrainScoreDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'shuffle': True,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 315)
        #TODO check data type

        self.assertEqual(results.metadata.query((0, 'learningData'))['dimension']['length'], 315)

        column_names = ['d3mIndex', 'timestamp', 'value_0', 'value_1', 'value_2', 'value_3', 'value_4','ground_truth']
        for i in range(8):
            self.assertEqual(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, i))['name'],
                             column_names[i])

        self.assertEqual(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, 0))['semantic_types'], (
            'http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/Index'
        ))
        for i in range(2, 6):
            self.assertEqual(
                results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, i))['semantic_types'], ('http://schema.org/Float',)
            )
        print(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, 7))['semantic_types'])
        self.assertEqual(results.metadata.query((0, 'learningData', metadata_base.ALL_ELEMENTS, 7))['semantic_types'], (
            'http://schema.org/Integer',
            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget',
        ))


if __name__ == '__main__':
    unittest.main()
