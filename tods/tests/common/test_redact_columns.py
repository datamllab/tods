import os
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.common import RedactColumns

class RedactColumnsPrimitiveTestCase(unittest.TestCase):

    def _get_datasets(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Index')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        datasets = container.List([dataset], {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
            'dimension': {
                'length': 1,
            },
        }, generate_metadata=False)

        # We update metadata based on metadata of each dataset.
        # TODO: In the future this might be done automatically by generate_metadata.
        #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/119
        for index, dataset in enumerate(datasets):
            datasets.metadata = dataset.metadata.copy_to(datasets.metadata, (), (index,))

        return dataset_doc_path, datasets

    def test_basic(self):
        dataset_doc_path, datasets = self._get_datasets()

        hyperparams_class = RedactColumns.RedactColumnsPrimitive.metadata.get_hyperparams()

        primitive = RedactColumns.RedactColumnsPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
            'add_semantic_types': ('https://metadata.datadrivendiscovery.org/types/RedactedTarget', 'https://metadata.datadrivendiscovery.org/types/MissingData'),
        }))
        redacted_datasets = primitive.produce(inputs=datasets).value

        self.assertTrue(len(redacted_datasets), 1)

        redacted_dataset = redacted_datasets[0]

        self.assertIsInstance(redacted_dataset, container.Dataset)

        # TODO: check metadata of yahoo dataset
        #self._test_metadata(redacted_datasets.metadata, dataset_doc_path, True)
        #self._test_metadata(redacted_dataset.metadata, dataset_doc_path, False)

    def _test_metadata(self, metadata, dataset_doc_path, is_list):
        top_metadata = {
            'structural_type': 'd3m.container.dataset.Dataset',
            'id': 'yahoo_sub_5_dataset_TRAIN',
            'version': '4.0.0',
            'name': 'Iris Dataset',
            'location_uris': [
                'file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path),
            ],
            'dimension': {
                'name': 'resources',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                'length': 1,
            },
            'digest': '49404bf166238fbdac2b6d6baa899a0d1bf8ed5976525fa7353fd732ac218a85',
            'source': {
                'license': 'CC',
                'redacted': False,
                'human_subjects_research': False,
            },
        }

        if is_list:
            prefix = [0]
            list_metadata = [{
                'selector': [],
                'metadata': {
                    'dimension': {
                        'length': 1,
                    },
                   'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                   'structural_type': 'd3m.container.list.List',
                },
            }]
        else:
            prefix = []
            list_metadata = []
            top_metadata['schema'] = metadata_base.CONTAINER_SCHEMA_VERSION

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), list_metadata + [{
            'selector': prefix + [],
            'metadata': top_metadata,
        }, {
            'selector': prefix + ['learningData'],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table', 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                },
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'structural_type': 'str',
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'sepalLength',
                'structural_type': 'str',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'sepalWidth',
                'structural_type': 'str',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'petalLength',
                'structural_type': 'str',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'petalWidth',
                'structural_type': 'str',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }, {
            'selector': prefix + ['learningData', '__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'species',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                    'https://metadata.datadrivendiscovery.org/types/RedactedTarget',
                    'https://metadata.datadrivendiscovery.org/types/MissingData',
                ],
            },
        }])


if __name__ == '__main__':
    unittest.main()
