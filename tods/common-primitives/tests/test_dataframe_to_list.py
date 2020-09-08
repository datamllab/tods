import unittest

from d3m import container

from common_primitives import dataframe_to_list, dataset_to_dataframe

import utils as test_utils


class DataFrameToListPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()

        # convert the dataset into a dataframe
        dataset_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        # convert the dataframe into a list
        list_hyperparams_class = dataframe_to_list.DataFrameToListPrimitive.metadata.get_hyperparams()
        list_primitive = dataframe_to_list.DataFrameToListPrimitive(hyperparams=list_hyperparams_class.defaults())
        list_value = list_primitive.produce(inputs=dataframe).value

        self.assertIsInstance(list_value, container.List)

        # verify dimensions
        self.assertEqual(len(list_value), 150)
        self.assertEqual(len(list_value[0]), 6)

        # verify data type is unchanged
        for row in list_value:
            for val in row:
                self.assertIsInstance(val, str)

        # validate metadata
        test_utils.test_iris_metadata(self, list_value.metadata, 'd3m.container.list.List', 'd3m.container.list.List')


if __name__ == '__main__':
    unittest.main()
