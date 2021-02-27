import os
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.common import Denormalize

from tods.tests import utils as test_utils


class DenormalizePrimitiveTestCase(unittest.TestCase):
    def _get_yahoo_dataset(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        return dataset

    def test_discard(self):

        dataset = self._get_yahoo_dataset() 

        dataset_metadata_before = dataset.metadata.to_internal_json_structure()

        hyperparams_class = Denormalize.DenormalizePrimitive.metadata.get_hyperparams()

        primitive = Denormalize.DenormalizePrimitive(hyperparams=hyperparams_class.defaults().replace({
            'recursive': False,
            'discard_not_joined_tabular_resources': True,
        }))

        denormalized_dataset = primitive.produce(inputs=dataset).value

        self.assertIsInstance(denormalized_dataset, container.Dataset)

        self.assertEqual(len(denormalized_dataset), 1)

        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 2]), {'AAA name', 'BBB name', 'CCC name'})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 3]), {'1', '2', ''})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 4]), {'aaa', 'bbb', 'ccc', 'ddd', 'eee'})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 5]), {'1990', '2000', '2010'})

        #self._test_discard_metadata(denormalized_dataset.metadata, dataset_doc_path)

        #self.assertEqual(dataset.metadata.to_internal_json_structure(), dataset_metadata_before)

    def _test_discard_metadata(self, metadata, dataset_doc_path):    # pragma: no cover
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.dataset.Dataset',
            'id': 'database_dataset_1',
            'version': '4.0.0',
            'name': 'A dataset simulating a database dump',
            'location_uris': [
                'file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path),
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/DatasetResource',
                ],
                'length': 1,
            },
            'digest': '68c435c6ba9a1c419c79507275c0d5710786dfe481e48f35591d87a7dbf5bb1a',
            'description': 'A synthetic dataset trying to be similar to a database dump, with tables with different relations between them.',
            'source': {
                'license': 'CC',
                'redacted': False,
            },
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData',))), {
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/TabularRow',
                ],
                'length': 45,
            },
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 7,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 3))), {
            'name': 'author',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            'foreign_key': {
                'type': 'COLUMN',
                'resource_id': 'authors',
                'column_index': 0,
            },
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))), {
            'name': 'code',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 2))), {
            'name': 'name',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Text',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 4))), {
            'name': 'key',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 5))), {
            'name': 'year',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/DateTime',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 6))), {
            'name': 'value',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

    def test_recursive(self):
        dataset = self._get_yahoo_dataset() 

        dataset_metadata_before = dataset.metadata.to_internal_json_structure()

        hyperparams_class = Denormalize.DenormalizePrimitive.metadata.get_hyperparams()

        primitive = Denormalize.DenormalizePrimitive(hyperparams=hyperparams_class.defaults().replace({
            'recursive': True,
            'discard_not_joined_tabular_resources': False,
        }))

        denormalized_dataset = primitive.produce(inputs=dataset).value

        self.assertIsInstance(denormalized_dataset, container.Dataset)

        self.assertEqual(len(denormalized_dataset), 1)

        #self.assertEqual(denormalized_dataset['values'].shape[0], 64)
        #self.assertEqual(denormalized_dataset['learningData'].shape[1], 8)

        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 1]), {'AAA', 'BBB', 'CCC'})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 2]), {'AAA name', 'BBB name', 'CCC name'})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 3]), {'1', '2', ''})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 4]), {'1 name', '2 name', ''})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 5]), {'aaa', 'bbb', 'ccc', 'ddd', 'eee'})
        #self.assertEqual(set(denormalized_dataset['learningData'].iloc[:, 6]), {'1990', '2000', '2010'})

        #self._test_recursive_metadata(denormalized_dataset.metadata, dataset_doc_path)

        #self.assertEqual(dataset.metadata.to_internal_json_structure(), dataset_metadata_before)

    def _test_recursive_metadata(self, metadata, dataset_doc_path):    # pragma: no cover
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.dataset.Dataset',
            'id': 'database_dataset_1',
            'version': '4.0.0',
            'name': 'A dataset simulating a database dump',
            'location_uris': [
                'file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path),
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/DatasetResource',
                ],
                'length': 4,
            },
            'digest': '68c435c6ba9a1c419c79507275c0d5710786dfe481e48f35591d87a7dbf5bb1a',
            'description': 'A synthetic dataset trying to be similar to a database dump, with tables with different relations between them.',
            'source': {
                'license': 'CC',
                'redacted': False,
            },
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData',))), {
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/TabularRow',
                ],
                'length': 45,
            },
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 8,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 3))), {
            'name': 'id',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))), {
            'name': 'code',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        for i in [2, 4]:
            self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))), {
                'name': ['name', None, 'name'][i - 2],
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Text',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            }, i)

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 5))), {
            'name': 'key',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 6))), {
            'name': 'year',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/DateTime',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 7))), {
            'name': 'value',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

    def test_row_order(self):
        dataset = self._get_yahoo_dataset()

        dataset_metadata_before = dataset.metadata.to_internal_json_structure()

        hyperparams_class = Denormalize.DenormalizePrimitive.metadata.get_hyperparams()

        primitive = Denormalize.DenormalizePrimitive(hyperparams=hyperparams_class.defaults().replace({
            'recursive': True,
            'discard_not_joined_tabular_resources': False,
        }))

        denormalized_dataset = primitive.produce(inputs=dataset).value

        self.assertIsInstance(denormalized_dataset, container.Dataset)

        self.assertEqual(len(denormalized_dataset), 1)

        #self.assertEqual(denormalized_dataset['learningData'].shape, (5, 3))

        #self.assertEqual(denormalized_dataset['learningData'].values.tolist(), [
        #    ['0', 'mnist_0_2.png', 'mnist'],
        #    ['1', 'mnist_1_1.png', 'mnist'],
        #    ['2', '001_HandPhoto_left_01.jpg', 'handgeometry'],
        #    ['3', 'cifar10_bird_1.png', 'cifar'],
        #    ['4', 'cifar10_bird_2.png', 'cifar'],
        #])

        #self._test_row_order_metadata(denormalized_dataset.metadata, dataset_doc_path)

        #self.assertEqual(dataset.metadata.to_internal_json_structure(), dataset_metadata_before)

    def _test_row_order_metadata(self, metadata, dataset_doc_path):    # pragma: no cover
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.dataset.Dataset',
            'id': 'image_dataset_1',
            'version': '4.0.0',
            'name': 'Image dataset to be used for tests',
            'location_uris': [
                'file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path),
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/DatasetResource',
                ],
                'length': 1,
            },
            'digest': '9b5553ce5ad84dfcefd379814dc6b11ef60a049479e3e91aa1251f7a5ef7409e',
            'description': 'There are a total of 5 image files, one is a left hand from the handgeometry dataset, two birds from cifar10 dataset and 2 figures from mnist dataset.',
            'source': {
                'license': 'Creative Commons Attribution-NonCommercial 4.0',
                'redacted': False,
            },
            'approximate_stored_size': 24000,
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData',))), {
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/TabularRow',
                ],
                'length': 5,
            },
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 3,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))), {
            'name': 'filename',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/FileName',
                'http://schema.org/ImageObject',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/UniqueKey',
            ],
            'location_base_uris': [
                'file://{dataset_base_path}/media/'.format(dataset_base_path=os.path.dirname(dataset_doc_path)),
            ],
            'media_types': [
                'image/jpeg',
                'image/png',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 2))), {
            'name': 'class',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', 0, 1))), {
            'name': 'filename',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/FileName',
                'http://schema.org/ImageObject',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/UniqueKey',
            ],
            'location_base_uris': [
                'file://{dataset_base_path}/media/'.format(dataset_base_path=os.path.dirname(dataset_doc_path)),
            ],
            'media_types': [
                'image/png',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query(('learningData', 2, 1))), {
            'name': 'filename',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/FileName',
                'http://schema.org/ImageObject',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/UniqueKey',
            ],
            'location_base_uris': [
                'file://{dataset_base_path}/media/'.format(dataset_base_path=os.path.dirname(dataset_doc_path)),
            ],
            'media_types': [
                'image/jpeg',
            ],
        })


if __name__ == '__main__':
    unittest.main()
