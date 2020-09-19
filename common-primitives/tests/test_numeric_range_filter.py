import unittest
import os

from common_primitives import numeric_range_filter
from d3m import container

import utils as test_utils


class NumericRangeFilterPrimitiveTestCase(unittest.TestCase):
    def test_inclusive_strict(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = numeric_range_filter.NumericRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'min': 6.5,
            'max': 6.7,
            'strict': True,
            'inclusive': True
        })
        filter_primitive = numeric_range_filter.NumericRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertGreater(new_dataframe['sepalLength'].astype(float).min(), 6.5)
        self.assertLess(new_dataframe['sepalLength'].astype(float).max(), 6.7)

    def test_inclusive_permissive(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = numeric_range_filter.NumericRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'min': 6.5,
            'max': 6.7,
            'strict': False,
            'inclusive': True
        })
        filter_primitive = numeric_range_filter.NumericRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertGreaterEqual(new_dataframe['sepalLength'].astype(float).min(), 6.5)
        self.assertLessEqual(new_dataframe['sepalLength'].astype(float).max(), 6.7)

    def test_exclusive_strict(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = numeric_range_filter \
            .NumericRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'min': 6.5,
            'max': 6.7,
            'strict': True,
            'inclusive': False
        })
        filter_primitive = numeric_range_filter.NumericRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertEqual(
            len(new_dataframe.loc[
                (new_dataframe['sepalLength'].astype(float) >= 6.5) &
                (new_dataframe['sepalLength'].astype(float) <= 6.7)]), 0)

    def test_exclusive_permissive(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = numeric_range_filter \
            .NumericRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'min': 6.5,
            'max': 6.7,
            'strict': False,
            'inclusive': False
        })
        filter_primitive = numeric_range_filter.NumericRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertEqual(
            len(new_dataframe.loc[
                (new_dataframe['sepalLength'].astype(float) > 6.5) &
                (new_dataframe['sepalLength'].astype(float) < 6.7)]), 0)

    def test_row_metadata_removal(self):
        # load the iris dataset
        dataset = test_utils.load_iris_metadata()

        # add metadata for rows 0 and 1
        dataset.metadata = dataset.metadata.update(('learningData', 0), {'a': 0})
        dataset.metadata = dataset.metadata.update(('learningData', 5), {'b': 1})

        resource = test_utils.get_dataframe(dataset)

        # apply filter that removes rows 0 and 1
        filter_hyperparams_class = numeric_range_filter.NumericRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 0,
            'min': 1,
            'max': 4,
            'strict': True,
            'inclusive': False
        })
        filter_primitive = numeric_range_filter.NumericRangeFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        # verify that the length is correct
        self.assertEqual(len(new_df), new_df.metadata.query(())['dimension']['length'])

        # verify that the rows were re-indexed in the metadata
        self.assertEqual(new_df.metadata.query((0,))['a'], 0)
        self.assertEqual(new_df.metadata.query((1,))['b'], 1)
        self.assertFalse('b' in new_df.metadata.query((5,)))

    def test_bad_type_handling(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = numeric_range_filter \
            .NumericRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'min': 6.5,
            'max': 6.7,
            'strict': False,
            'inclusive': False
        })
        filter_primitive = numeric_range_filter.NumericRangeFilterPrimitive(hyperparams=hp)
        with self.assertRaises(ValueError):
            filter_primitive.produce(inputs=resource)


if __name__ == '__main__':
    unittest.main()
