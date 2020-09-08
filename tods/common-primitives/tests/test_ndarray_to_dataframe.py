import unittest

import numpy

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataframe_to_ndarray, dataset_to_dataframe, ndarray_to_dataframe

import utils as test_utils


class NDArrayToDataFramePrimitiveTestCase(unittest.TestCase):
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

        # convert the numpy array back into a dataframe
        dataframe_hyperparams_class = ndarray_to_dataframe.NDArrayToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = ndarray_to_dataframe.NDArrayToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=numpy_array).value

        self.assertIsInstance(dataframe, container.DataFrame)

        # verify dimensions
        self.assertEqual(len(dataframe), 150)
        self.assertEqual(len(dataframe.iloc[0]), 6)

        # ensure column names added to dataframe
        self.assertListEqual(list(dataframe.columns.values), ['d3mIndex', 'sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species'])

        # verify data type is unchanged
        for row in dataframe:
            for cell in row:
                self.assertIsInstance(cell, str)

        # validate metadata
        test_utils.test_iris_metadata(self, dataframe.metadata, 'd3m.container.pandas.DataFrame')

    def test_vector(self):
        data = container.ndarray(numpy.array([1, 2, 3]), generate_metadata=True)

        dataframe_hyperparams_class = ndarray_to_dataframe.NDArrayToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = ndarray_to_dataframe.NDArrayToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=data).value

        self._test_vector_metadata(dataframe.metadata, True)

    def _test_vector_metadata(self, metadata, use_individual_columns):
        self.maxDiff = None

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
                'structural_type': 'numpy.int64',
            },
        }]

        if use_individual_columns:
            expected_metadata[-1]['selector'] = ['__ALL_ELEMENTS__', 0]

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), expected_metadata)


if __name__ == '__main__':
    unittest.main()
