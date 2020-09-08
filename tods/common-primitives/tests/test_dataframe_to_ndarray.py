import unittest

from common_primitives import dataframe_to_ndarray, dataset_to_dataframe
from d3m import container

import utils as test_utils


class DataFrameToNDArrayPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()

        # convert the dataset into a dataframe
        dataset_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        # convert the dataframe into a numpy array
        numpy_hyperparams_class = dataframe_to_ndarray.DataFrameToNDArrayPrimitive.metadata.get_hyperparams()
        numpy_primitive = dataframe_to_ndarray.DataFrameToNDArrayPrimitive(hyperparams=numpy_hyperparams_class.defaults())
        numpy_array = numpy_primitive.produce(inputs=dataframe).value

        self.assertIsInstance(numpy_array, container.ndarray)

        # verify dimensions
        self.assertEqual(len(numpy_array), 150)
        self.assertEqual(len(numpy_array[0]), 6)

        # verify data type is unchanged
        for row in numpy_array:
            for val in row:
                self.assertIsInstance(val, str)

        # validate metadata
        test_utils.test_iris_metadata(self, numpy_array.metadata, 'd3m.container.numpy.ndarray')


if __name__ == '__main__':
    unittest.main()
