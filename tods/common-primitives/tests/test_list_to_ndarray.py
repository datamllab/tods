import unittest

import numpy

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import list_to_ndarray


class ListToNDRrrayPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        data = container.List([container.List([1, 2, 3]), container.List([4, 5, 6])], generate_metadata=True)

        list_hyperparams_class = list_to_ndarray.ListToNDArrayPrimitive.metadata.get_hyperparams()
        list_primitive = list_to_ndarray.ListToNDArrayPrimitive(hyperparams=list_hyperparams_class.defaults())
        array = list_primitive.produce(inputs=data).value

        self._test_basic_metadata(array.metadata, 'numpy.int64')

    def _test_basic_metadata(self, metadata, structural_type):
        self.maxDiff = None

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
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
                'structural_type': structural_type,
            },
        }])

    def test_just_list(self):
        data = container.List([1, 2, 3], generate_metadata=True)

        list_hyperparams_class = list_to_ndarray.ListToNDArrayPrimitive.metadata.get_hyperparams()
        list_primitive = list_to_ndarray.ListToNDArrayPrimitive(hyperparams=list_hyperparams_class.defaults())
        array = list_primitive.produce(inputs=data).value

        self._test_just_list_metadata(array.metadata, 'numpy.int64')

    def _test_just_list_metadata(self, metadata, structural_type):
        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()),[{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': structural_type,
            },
        }])

    def test_list_ndarray(self):
        data = container.List([container.ndarray(numpy.array([[1, 2, 3], [11, 12, 13]])), container.ndarray(numpy.array([[4, 5, 6], [14, 15, 16]]))], generate_metadata=True)

        list_hyperparams_class = list_to_ndarray.ListToNDArrayPrimitive.metadata.get_hyperparams()
        list_primitive = list_to_ndarray.ListToNDArrayPrimitive(hyperparams=list_hyperparams_class.defaults())
        array = list_primitive.produce(inputs=data).value

        self._test_list_ndarray_metadata(array.metadata)

    def _test_list_ndarray_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'length': 2,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'semantic_types': '__NO_VALUE__',
                'dimension': {
                    'length': 2,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 3,
                    'semantic_types': '__NO_VALUE__',
                    'name': '__NO_VALUE__',
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])


if __name__ == '__main__':
    unittest.main()
