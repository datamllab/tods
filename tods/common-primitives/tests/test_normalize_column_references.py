import os
import unittest

from d3m import container, utils

from common_primitives import normalize_column_references

import utils as test_utils


class NormalizeColumnReferencesPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json')
        )

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        metadata_before = dataset.metadata.to_internal_json_structure()

        self._test_metadata_before(utils.to_json_structure(dataset.metadata.to_internal_simple_structure()), dataset_doc_path)

        hyperparams_class = normalize_column_references.NormalizeColumnReferencesPrimitive.metadata.get_hyperparams()

        primitive = normalize_column_references.NormalizeColumnReferencesPrimitive(
            hyperparams=hyperparams_class.defaults()
        )

        normalized_dataset = primitive.produce(inputs=dataset).value

        self.assertIsInstance(normalized_dataset, container.Dataset)

        self._test_metadata_after(utils.to_json_structure(normalized_dataset.metadata.to_internal_simple_structure()), dataset_doc_path)

        self.assertEqual(metadata_before, dataset.metadata.to_internal_json_structure())

    def _test_metadata_before(self, metadata, dataset_doc_path):
        self.maxDiff = None

        self.assertEqual(
            test_utils.convert_through_json(metadata),
            [
                {
                    'selector': [],
                    'metadata': {
                        'description': 'A synthetic dataset trying to be similar to a database dump, with tables with different relations between them.',
                        'digest': '68c435c6ba9a1c419c79507275c0d5710786dfe481e48f35591d87a7dbf5bb1a',
                        'dimension': {
                            'length': 4,
                            'name': 'resources',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                        },
                        'id': 'database_dataset_1',
                        'location_uris': [
                            'file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path),
                        ],
                        'name': 'A dataset simulating a database dump',
                        'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                        'source': {'license': 'CC', 'redacted': False},
                        'structural_type': 'd3m.container.dataset.Dataset',
                        'version': '4.0.0',
                    },
                },
                {
                    'selector': ['authors'],
                    'metadata': {
                        'dimension': {
                            'length': 3,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['authors', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 2,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['authors', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'name': 'id',
                        'semantic_types': [
                            'http://schema.org/Integer',
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['authors', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'name': 'name',
                        'semantic_types': [
                            'http://schema.org/Text',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['codes'],
                    'metadata': {
                        'dimension': {
                            'length': 3,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 3,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'name': 'code',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'name': 'name',
                        'semantic_types': [
                            'http://schema.org/Text',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__', 2],
                    'metadata': {
                        'foreign_key': {'column_index': 0, 'resource_id': 'authors', 'type': 'COLUMN'},
                        'name': 'author',
                        'semantic_types': [
                            'http://schema.org/Integer',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData'],
                    'metadata': {
                        'dimension': {
                            'length': 45,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/Table',
                            'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                        ],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 5,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'name': 'd3mIndex',
                        'semantic_types': [
                            'http://schema.org/Integer',
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'foreign_key': {'column_name': 'code', 'resource_id': 'codes', 'type': 'COLUMN'},
                        'name': 'code',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 2],
                    'metadata': {
                        'name': 'key',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 3],
                    'metadata': {
                        'name': 'year',
                        'semantic_types': [
                            'http://schema.org/DateTime',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 4],
                    'metadata': {
                        'name': 'value',
                        'semantic_types': [
                            'http://schema.org/Float',
                            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values'],
                    'metadata': {
                        'dimension': {
                            'length': 64,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 4,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'foreign_key': {'column_name': 'code', 'resource_id': 'codes', 'type': 'COLUMN'},
                        'name': 'code',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'name': 'key',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 2],
                    'metadata': {
                        'name': 'year',
                        'semantic_types': [
                            'http://schema.org/DateTime',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 3],
                    'metadata': {
                        'name': 'value',
                        'semantic_types': [
                            'http://schema.org/Float',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
            ],
        )

    def _test_metadata_after(self, metadata, dataset_doc_path):
        self.maxDiff = None

        self.assertEqual(
            test_utils.convert_through_json(metadata),
            [
                {
                    'selector': [],
                    'metadata': {
                        'description': 'A synthetic dataset trying to be similar to a database dump, with tables with different relations between them.',
                        'digest': '68c435c6ba9a1c419c79507275c0d5710786dfe481e48f35591d87a7dbf5bb1a',
                        'dimension': {
                            'length': 4,
                            'name': 'resources',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                        },
                        'id': 'database_dataset_1',
                        'location_uris': [
                            'file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path),
                        ],
                        'name': 'A dataset simulating a database dump',
                        'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                        'source': {'license': 'CC', 'redacted': False},
                        'structural_type': 'd3m.container.dataset.Dataset',
                        'version': '4.0.0',
                    },
                },
                {
                    'selector': ['authors'],
                    'metadata': {
                        'dimension': {
                            'length': 3,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['authors', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 2,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['authors', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'name': 'id',
                        'semantic_types': [
                            'http://schema.org/Integer',
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['authors', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'name': 'name',
                        'semantic_types': [
                            'http://schema.org/Text',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['codes'],
                    'metadata': {
                        'dimension': {
                            'length': 3,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 3,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'name': 'code',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'name': 'name',
                        'semantic_types': [
                            'http://schema.org/Text',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['codes', '__ALL_ELEMENTS__', 2],
                    'metadata': {
                        'foreign_key': {'column_index': 0, 'resource_id': 'authors', 'type': 'COLUMN'},
                        'name': 'author',
                        'semantic_types': [
                            'http://schema.org/Integer',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData'],
                    'metadata': {
                        'dimension': {
                            'length': 45,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/Table',
                            'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                        ],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 5,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'name': 'd3mIndex',
                        'semantic_types': [
                            'http://schema.org/Integer',
                            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'foreign_key': {'column_index': 0, 'column_name': '__NO_VALUE__', 'resource_id': 'codes', 'type': 'COLUMN'},
                        'name': 'code',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 2],
                    'metadata': {
                        'name': 'key',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 3],
                    'metadata': {
                        'name': 'year',
                        'semantic_types': [
                            'http://schema.org/DateTime',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['learningData', '__ALL_ELEMENTS__', 4],
                    'metadata': {
                        'name': 'value',
                        'semantic_types': [
                            'http://schema.org/Float',
                            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values'],
                    'metadata': {
                        'dimension': {
                            'length': 64,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': 'd3m.container.pandas.DataFrame',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__'],
                    'metadata': {
                        'dimension': {
                            'length': 4,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 0],
                    'metadata': {
                        'foreign_key': {'column_index': 0, 'column_name': '__NO_VALUE__', 'resource_id': 'codes', 'type': 'COLUMN'},
                        'name': 'code',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 1],
                    'metadata': {
                        'name': 'key',
                        'semantic_types': [
                            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 2],
                    'metadata': {
                        'name': 'year',
                        'semantic_types': [
                            'http://schema.org/DateTime',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
                {
                    'selector': ['values', '__ALL_ELEMENTS__', 3],
                    'metadata': {
                        'name': 'value',
                        'semantic_types': [
                            'http://schema.org/Float',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                        ],
                        'structural_type': 'str',
                    },
                },
            ],
        )


if __name__ == '__main__':
    unittest.main()
