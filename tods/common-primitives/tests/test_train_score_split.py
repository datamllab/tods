import os
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import train_score_split


class TrainScoreDatasetSplitPrimitiveTestCase(unittest.TestCase):
    def test_produce_train(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = train_score_split.TrainScoreDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = train_score_split.TrainScoreDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
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

        self.assertEqual(results[0]['learningData'].shape[0], 112)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [
            '0', '1', '2', '3', '4', '5', '6', '9', '10', '11', '12', '13', '14', '15', '17', '19', '20',
            '21', '23', '25', '28', '29', '30', '31', '32', '34', '35', '36', '38', '39', '41', '42', '43',
            '46', '47', '48', '49', '50', '52', '53', '55', '56', '57', '58', '60', '61', '64', '65', '67',
            '68', '69', '70', '72', '74', '75', '77', '79', '80', '81', '82', '85', '87', '88', '89', '91',
            '92', '94', '95', '96', '98', '99', '101', '102', '103', '104', '105', '106', '108', '109', '110',
            '111', '112', '113', '115', '116', '117', '118', '119', '120', '122', '123', '124', '125', '128',
            '129', '130', '131', '133', '135', '136', '138', '139', '140', '141', '142', '143', '144', '145',
            '146', '147', '148', '149',
        ])

    def test_produce_score(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = train_score_split.TrainScoreDatasetSplitPrimitive.metadata.get_hyperparams()

        primitive = train_score_split.TrainScoreDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'shuffle': True,
        }))

        primitive.set_training_data(dataset=dataset)
        primitive.fit()

        results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

        self.assertEqual(len(results), 1)

        for dataset in results:
            self.assertEqual(len(dataset), 1)

        self.assertEqual(results[0]['learningData'].shape[0], 38)
        self.assertEqual(list(results[0]['learningData'].iloc[:, 0]), [
            '7', '8', '16', '18', '22', '24', '26', '27', '33', '37', '40', '44', '45', '51', '54',
            '59', '62', '63', '66', '71', '73', '76', '78', '83', '84', '86', '90', '93', '97', '100',
            '107', '114', '121', '126', '127', '132', '134', '137',
        ])


if __name__ == '__main__':
    unittest.main()
