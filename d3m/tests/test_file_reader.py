import unittest
import os.path
import sys

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')
sys.path.insert(0, TEST_PRIMITIVES_DIR)

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

from test_primitives.file_reader import DummyImageReaderPrimitive

from d3m import container, utils


class TestDummyImageReaderPrimitive(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'image_dataset_1', 'datasetDoc.json')
        )

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = DatasetToDataFramePrimitive(
            hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'})
        )
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        image_hyperparams_class = DummyImageReaderPrimitive.metadata.get_hyperparams()
        image_primitive = DummyImageReaderPrimitive(
            hyperparams=image_hyperparams_class.defaults().replace({'return_result': 'replace'})
        )
        images_names = image_primitive.produce(inputs=dataframe).value

        self.assertEqual(images_names.iloc[0]['filename'][0], '001_HandPhoto_left_01.jpg')
        self.assertEqual(images_names.iloc[1]['filename'][0], 'cifar10_bird_1.png')
        self.assertEqual(images_names.iloc[2]['filename'][0], 'cifar10_bird_2.png')
        self.assertEqual(images_names.iloc[3]['filename'][0], 'mnist_0_2.png')
        self.assertEqual(images_names.iloc[4]['filename'][0], 'mnist_1_1.png')

        self._test_metadata(images_names.metadata)

    def _test_metadata(self, metadata):
        self.assertEqual(
            utils.to_json_structure(metadata.to_internal_simple_structure()),
            [
                {
                    'metadata': {
                        'dimension': {
                            'length': 5,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/Table',
                            'https://metadata.datadrivendiscovery.org/types/FilesCollection',
                        ],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                    'selector': [],
                },
                {
                    'metadata': {
                        'dimension': {
                            'length': 1,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                    'selector': ['__ALL_ELEMENTS__'],
                },
                {
                    'metadata': {
                        'dimension': {
                            'length': 1,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'location_base_uris': '__NO_VALUE__',
                        'media_types': '__NO_VALUE__',
                        'name': 'filename',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'http://schema.org/ImageObject',
                            'https://metadata.datadrivendiscovery.org/types/Table',
                        ],
                        'structural_type': 'd3m.container.numpy.ndarray',
                    },
                    'selector': ['__ALL_ELEMENTS__', 0],
                },
                {
                    'metadata': {
                        'dimension': {
                            'length': 1,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                    'selector': ['__ALL_ELEMENTS__', 0, '__ALL_ELEMENTS__'],
                },
                {
                    'metadata': {'structural_type': 'str'},
                    'selector': ['__ALL_ELEMENTS__', 0, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                },
                {
                    'metadata': {
                        'image_reader_metadata': {'foobar': 42},
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'http://schema.org/ImageObject',
                            'https://metadata.datadrivendiscovery.org/types/Table',
                        ],
                    },
                    'selector': [0, 0],
                },
                {
                    'metadata': {
                        'image_reader_metadata': {'foobar': 42},
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'http://schema.org/ImageObject',
                            'https://metadata.datadrivendiscovery.org/types/Table',
                        ],
                    },
                    'selector': [1, 0],
                },
                {
                    'metadata': {
                        'image_reader_metadata': {'foobar': 42},
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'http://schema.org/ImageObject',
                            'https://metadata.datadrivendiscovery.org/types/Table',
                        ],
                    },
                    'selector': [2, 0],
                },
                {
                    'metadata': {
                        'image_reader_metadata': {'foobar': 42},
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'http://schema.org/ImageObject',
                            'https://metadata.datadrivendiscovery.org/types/Table',
                        ],
                    },
                    'selector': [3, 0],
                },
                {
                    'metadata': {
                        'image_reader_metadata': {'foobar': 42},
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                            'http://schema.org/ImageObject',
                            'https://metadata.datadrivendiscovery.org/types/Table',
                        ],
                    },
                    'selector': [4, 0],
                },
            ],
        )


if __name__ == '__main__':
    unittest.main()
