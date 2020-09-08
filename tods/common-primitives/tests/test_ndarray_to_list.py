import unittest

import numpy

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataframe_to_ndarray, dataset_to_dataframe, ndarray_to_list

import utils as test_utils


class NDArrayToListPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        # TODO: Find a less cumbersome way to get a numpy array loaded with a dataset
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()

        # convert the dataset into a dataframe
        dataset_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataset_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_hyperparams_class.defaults())
        dataframe_dataset = dataset_primitive.produce(inputs=dataset).value

        # convert the dataframe into a numpy array
        numpy_hyperparams_class = dataframe_to_ndarray.DataFrameToNDArrayPrimitive.metadata.get_hyperparams()
        numpy_primitive = dataframe_to_ndarray.DataFrameToNDArrayPrimitive(hyperparams=numpy_hyperparams_class.defaults())
        numpy_array = numpy_primitive.produce(inputs=dataframe_dataset).value

        list_hyperparams_class = ndarray_to_list.NDArrayToListPrimitive.metadata.get_hyperparams()
        list_primitive = ndarray_to_list.NDArrayToListPrimitive(hyperparams=list_hyperparams_class.defaults())
        list_value = list_primitive.produce(inputs=numpy_array).value

        self.assertIsInstance(list_value, container.List)

        # verify dimensions
        self.assertEqual(len(list_value), 150)
        self.assertEqual(len(list_value[0]), 6)

        # validate metadata
        test_utils.test_iris_metadata(self, list_value.metadata, 'd3m.container.list.List', 'd3m.container.numpy.ndarray')

    def test_vector(self):
        data = container.ndarray(numpy.array([1, 2, 3]), generate_metadata=True)

        list_hyperparams_class = ndarray_to_list.NDArrayToListPrimitive.metadata.get_hyperparams()
        list_primitive = ndarray_to_list.NDArrayToListPrimitive(hyperparams=list_hyperparams_class.defaults())
        list_value = list_primitive.produce(inputs=data).value

        self._test_vector_metadata(list_value.metadata)

    def _test_vector_metadata(self, metadata):
        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

    def test_deep_array(self):
        data = container.ndarray(numpy.array(range(2 * 3 * 4)).reshape((2, 3, 4)), generate_metadata=True)

        list_hyperparams_class = ndarray_to_list.NDArrayToListPrimitive.metadata.get_hyperparams()
        list_primitive = ndarray_to_list.NDArrayToListPrimitive(hyperparams=list_hyperparams_class.defaults())
        list_value = list_primitive.produce(inputs=data).value

        self._test_deep_vector_metadata(list_value.metadata)

    def _test_deep_vector_metadata(self, metadata):
        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'structural_type': 'd3m.container.numpy.ndarray',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
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
