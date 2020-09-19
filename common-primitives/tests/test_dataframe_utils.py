import unittest
import os

from common_primitives import dataframe_utils
from d3m import container
from d3m.base import utils as base_utils

import utils as test_utils


class DataFrameUtilsTestCase(unittest.TestCase):
    def test_inclusive(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        to_keep_indices = [1, 2, 5]

        output = dataframe_utils.select_rows(resource, to_keep_indices)
        self.assertEqual(len(output), 3)
        self.assertEqual(len(output.iloc[0]), 5)
        self.assertEqual(output.iloc[1, 0], '3')
        self.assertEqual(output.iloc[2, 0], '6')


if __name__ == '__main__':
    unittest.main()
