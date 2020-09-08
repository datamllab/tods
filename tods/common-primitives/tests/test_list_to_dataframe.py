import unittest

import numpy

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import list_to_dataframe


class ListToDataFramePrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        data = container.List([container.List([1, 2, 3]), container.List([4, 5, 6])], generate_metadata=True)

        list_hyperparams_class = list_to_dataframe.ListToDataFramePrimitive.metadata.get_hyperparams()
        list_primitive = list_to_dataframe.ListToDataFramePrimitive(hyperparams=list_hyperparams_class.defaults())
        dataframe = list_primitive.produce(inputs=data).value

        self._test_basic_metadata(dataframe.metadata, 'numpy.int64', True)

    def _test_basic_metadata(self, metadata, structural_type, add_individual_columns):
        expected_metadata = [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'dimension': {
                    'length': 2,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 3,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }]

        if add_individual_columns:
            expected_metadata.extend([{
                'selector': ['__ALL_ELEMENTS__', 0],
                'metadata': {
                    'structural_type': structural_type,
                },
            }, {
                'selector': ['__ALL_ELEMENTS__', 1],
                'metadata': {
                    'structural_type': structural_type,
                },
            }, {
                'selector': ['__ALL_ELEMENTS__', 2],
                'metadata': {
                    'structural_type': structural_type,
                },
            }])

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), expected_metadata)

    def test_just_list(self):
        data = container.List([1, 2, 3], generate_metadata=True)

        list_hyperparams_class = list_to_dataframe.ListToDataFramePrimitive.metadata.get_hyperparams()
        list_primitive = list_to_dataframe.ListToDataFramePrimitive(hyperparams=list_hyperparams_class.defaults())
        dataframe = list_primitive.produce(inputs=data).value

        self._test_just_list_metadata(dataframe.metadata, 'numpy.int64', True)

    def _test_just_list_metadata(self, metadata, structural_type, use_individual_columns):
        expected_metadata = [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 1,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': structural_type,
            },
        }]

        if use_individual_columns:
            expected_metadata[-1]['selector'] = ['__ALL_ELEMENTS__', 0]

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), expected_metadata)

    def test_list_ndarray(self):
        data = container.List([container.ndarray(numpy.array([1, 2, 3])), container.ndarray(numpy.array([4, 5, 6]))], generate_metadata=True)

        list_hyperparams_class = list_to_dataframe.ListToDataFramePrimitive.metadata.get_hyperparams()
        list_primitive = list_to_dataframe.ListToDataFramePrimitive(hyperparams=list_hyperparams_class.defaults())
        dataframe = list_primitive.produce(inputs=data).value

        self._test_list_ndarray_metadata(dataframe.metadata, True)

    def _test_list_ndarray_metadata(self, metadata, add_individual_columns):
        expected_metadata = [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'dimension': {
                    'length': 2,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 3,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }]

        if add_individual_columns:
            expected_metadata.extend([{
                'selector': ['__ALL_ELEMENTS__', 0],
                'metadata': {
                    'structural_type': 'numpy.int64',
                },
            }, {
                'selector': ['__ALL_ELEMENTS__', 1],
                'metadata': {
                    'structural_type': 'numpy.int64',
                },
            }, {
                'selector': ['__ALL_ELEMENTS__', 2],
                'metadata': {
                    'structural_type': 'numpy.int64',
                },
            }])

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), expected_metadata)

    def test_list_deeper_ndarray(self):
        data = container.List([container.ndarray(numpy.array([[1, 2, 3], [11, 12, 13]])), container.ndarray(numpy.array([[4, 5, 6], [14, 15, 16]]))], generate_metadata=True)

        list_hyperparams_class = list_to_dataframe.ListToDataFramePrimitive.metadata.get_hyperparams()
        list_primitive = list_to_dataframe.ListToDataFramePrimitive(hyperparams=list_hyperparams_class.defaults())

        with self.assertRaisesRegex(ValueError, 'Must pass 2-d input'):
            list_primitive.produce(inputs=data).value


if __name__ == '__main__':
    unittest.main()
