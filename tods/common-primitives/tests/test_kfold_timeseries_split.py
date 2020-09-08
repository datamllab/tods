import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import kfold_split_timeseries


class KFoldTimeSeriesSplitPrimitiveTestCase(unittest.TestCase):
    def test_produce_train_timeseries_1(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
             self.assertEqual(len(dataset), 1)

        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 8)
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'2013-11-05', '2013-11-06', '2013-11-07', '2013-11-08', '2013-11-11',
                '2013-11-12', '2013-11-13', '2013-11-14'})

        self.assertEqual(len(results[1]['learningData'].iloc[:, 0]), 8)
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'2013-11-13', '2013-11-14', '2013-11-15', '2013-11-18', '2013-11-19',
                '2013-11-20', '2013-11-21', '2013-11-22'})

    def test_produce_score_timeseries_1(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
             self.assertEqual(len(dataset), 1)

        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 6)
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'2013-11-15', '2013-11-18', '2013-11-19',
                '2013-11-20', '2013-11-21', '2013-11-22'})

        self.assertEqual(len(results[1]['learningData'].iloc[:, 0]), 6)
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'2013-11-25', '2013-11-26', '2013-11-27',
                '2013-11-29', '2013-12-02', '2013-12-03'})

    def test_produce_train(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        # We fake that the dataset is time-series.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Time')

        hyperparams_class = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
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
        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 9)
        self.assertEqual(set(results[0]['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 2]), {'bbb', 'ccc', 'ddd'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'1990'})

        self.assertEqual(set(results[1]['codes'].iloc[:, 0]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(len(results[1]['learningData'].iloc[:, 0]), 9)
        self.assertEqual(set(results[1]['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 2]), {'aaa', 'bbb', 'ddd', 'eee'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'1990', '2000'})

    def test_produce_score(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        # We fake that the dataset is time-series.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 3), 'https://metadata.datadrivendiscovery.org/types/Time')

        hyperparams_class = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
            self.assertEqual(len(dataset), 4)

        self.assertEqual(results[0]['codes'].shape[0], 3)
        self.assertEqual(results[1]['codes'].shape[0], 3)

        self.assertEqual(set(results[0]['codes'].iloc[:, 0]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 0]), {'2', '3', '32', '33', '37', '38', '39'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 2]), {'aaa', 'ddd', 'eee'})
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'1990', '2000'})

        self.assertEqual(set(results[1]['codes'].iloc[:, 0]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 0]), {'22', '23', '24', '31', '40', '41', '42'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 2]), {'ccc', 'ddd', 'eee'})
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'2000'})

    def test_unsorted_datetimes_timeseries_4(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'timeseries_dataset_4', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive.metadata.get_hyperparams()

        folds = 5
        primitive = kfold_split_timeseries.KFoldTimeSeriesSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'number_of_folds': folds,
            'number_of_window_folds': 1,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        # To test that pickling works.
        pickle.dumps(primitive)

        results = primitive.produce(inputs=container.List([0, 1], generate_metadata=True)).value

        self.assertEqual(len(results), 2)

        for dataset in results:
             self.assertEqual(len(dataset), 1)

        self.assertEqual(len(results[0]['learningData'].iloc[:, 0]), 8)
        self.assertEqual(set(results[0]['learningData'].iloc[:, 3]), {'2013-11-05', '2013-11-06', '2013-11-07', '2013-11-08', '2013-11-11',
                '2013-11-12', '2013-11-13', '2013-11-14'})

        self.assertEqual(len(results[1]['learningData'].iloc[:, 0]), 8)
        self.assertEqual(set(results[1]['learningData'].iloc[:, 3]), {'2013-11-13', '2013-11-14', '2013-11-15', '2013-11-18', '2013-11-19',
                '2013-11-20', '2013-11-21', '2013-11-22'})


if __name__ == '__main__':
    unittest.main()
