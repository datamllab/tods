import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from tods.common import KFoldSplitTimeseries


class KFoldTimeSeriesSplitPrimitiveTestCase(unittest.TestCase):

    def _get_yahoo_dataset(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        return dataset

    def test_produce_train_timeseries_1(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Time')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
             self.assertEqual(len(dataset), 1)

        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 210)
        #TODO: correct the semantic type and validate unix timestamp

    def test_produce_score_timeseries_1(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Time')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
             self.assertEqual(len(dataset), 1)

        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 210)

    def test_produce_train(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Time')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        # We fake that the dataset is time-series.
        #dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Time')

        hyperparams_class = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 210)
        #TODO: correct the semantic type and validate unix timestamp

    def test_produce_score(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Time')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')


        hyperparams_class = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 210)


    def test_unsorted_datetimes_timeseries_4(self):
        dataset = self._get_yahoo_dataset()

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/Time')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = KFoldSplitTimeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
             self.assertEqual(len(dataset), 1)

        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 210)

        #TODO: correct the semantic type and validate unix timestamp


if __name__ == '__main__':
    unittest.main()
