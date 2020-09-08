import math
import os.path
import unittest

import numpy

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, column_parser, utils as common_utils

import utils as test_utils


class ColumnParserPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, (0, 5.1, 3.5, 1.4, 0.2, 6241605690342144121))

        self.assertEqual([type(o) for o in first_row], [int, float, float, float, float, int])

        self._test_basic_metadata(dataframe.metadata)

    def _test_basic_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 150,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 6,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'int',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        for i in range(1, 5):
            self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, i))), {
                'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'][i - 1],
                'structural_type': 'float',
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            }, i)

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))), {
            'name': 'species',
            'structural_type': 'int',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

    def test_new(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'use_columns': [2]}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, ('0', 3.5))

        self.assertEqual([type(o) for o in first_row], [str, float])

        self._test_new_metadata(dataframe.metadata)

    def _test_new_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 150,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 2,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {
            'name': 'sepalWidth',
            'structural_type': 'float',
            'semantic_types': [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

    def test_append(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'append', 'replace_index_columns': False, 'parse_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'http://schema.org/Integer']}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, ('0', '5.1', '3.5', '1.4', '0.2', 'Iris-setosa', 0, 6241605690342144121))

        self.assertEqual([type(o) for o in first_row], [str, str, str, str, str, str, int, int])

        self._test_append_metadata(dataframe.metadata, False)

    def test_append_replace_index_columns(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'append', 'parse_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'http://schema.org/Integer']}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, (0, '5.1', '3.5', '1.4', '0.2', 'Iris-setosa', 6241605690342144121))

        self.assertEqual([type(o) for o in first_row], [int, str, str, str, str, str, int])

        self._test_append_metadata(dataframe.metadata, True)

    def _test_append_metadata(self, metadata, replace_index_columns):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 150,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 7 if replace_index_columns else 8,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'int' if replace_index_columns else 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        for i in range(1, 5):
            self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, i))), {
                'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'][i - 1],
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            }, i)

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))), {
            'name': 'species',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        if not replace_index_columns:
            self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 6))), {
                'name': 'd3mIndex',
                'structural_type': 'int',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 6 if replace_index_columns else 7))), {
            'name': 'species',
            'structural_type': 'int',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

    def test_integer(self):
        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())

        dataframe = container.DataFrame({'a': ['1.0', '2.0', '3.0']}, generate_metadata=True)

        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0), {
            'name': 'test',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        call_metadata = primitive.produce(inputs=dataframe)

        parsed_dataframe = call_metadata.value

        self.assertEqual(test_utils.convert_through_json(parsed_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'test',
            'structural_type': 'int',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(list(parsed_dataframe.iloc[:, 0]), [1, 2, 3])

        dataframe.iloc[2, 0] = '3.1'

        call_metadata = primitive.produce(inputs=dataframe)

        parsed_dataframe = call_metadata.value

        self.assertEqual(test_utils.convert_through_json(parsed_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'test',
            'structural_type': 'int',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(list(parsed_dataframe.iloc[:, 0]), [1, 2, 3])

        dataframe.iloc[2, 0] = 'aaa'

        with self.assertRaisesRegex(ValueError, 'Not all values in a column can be parsed into integers, but only integers were expected'):
            primitive.produce(inputs=dataframe)

        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0), {
            'name': 'test',
            'structural_type': str,
            'semantic_types': [
                'http://schema.org/Integer',
            ],
        })

        call_metadata = primitive.produce(inputs=dataframe)

        parsed_dataframe = call_metadata.value

        self.assertEqual(test_utils.convert_through_json(parsed_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'test',
            'structural_type': 'float',
            'semantic_types': [
                'http://schema.org/Integer',
            ],
        })

        self.assertEqual(list(parsed_dataframe.iloc[0:2, 0]), [1.0, 2.0])
        self.assertTrue(math.isnan(parsed_dataframe.iloc[2, 0]))

    def test_float_vector(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'object_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults().replace({'dataframe_resource': 'learningData'}))
        dataframe = primitive.produce(inputs=dataset).value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        self.assertIsInstance(dataframe.iloc[0, 3], container.ndarray)
        self.assertEqual(dataframe.iloc[0, 3].shape, (8,))

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 4,
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json'},
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'structural_type': 'int',
                'name': 'd3mIndex',
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'image',
                'structural_type': 'str',
                'semantic_types': ['http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'foreign_key': {
                    'type': 'COLUMN',
                    'resource_id': '0',
                    'column_index': 0,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'color_not_class',
                'structural_type': 'int',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 8,
                },
                'name': 'bounding_polygon_area',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/FloatVector',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Boundary',
                    'https://metadata.datadrivendiscovery.org/types/BoundingPolygon',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
                'boundary_for': {
                    'resource_id': 'learningData',
                    'column_name': 'image',
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3, '__ALL_ELEMENTS__'],
            'metadata': {'structural_type': 'numpy.float64'},
        }])

    def test_ugly_time_values(self):
        for value in [
            'Original chained constant price data are rescaled.',
            '1986/87',
        ]:
            self.assertTrue(numpy.isnan(common_utils.parse_datetime_to_float(value)), value)


if __name__ == '__main__':
    unittest.main()
