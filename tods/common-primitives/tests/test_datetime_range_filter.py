import unittest
import os

from datetime import datetime
from dateutil import parser
from common_primitives import datetime_range_filter
from d3m import container

import utils as test_utils


class DatetimeRangeFilterPrimitiveTestCase(unittest.TestCase):
    def test_inclusive_strict(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = datetime_range_filter.DatetimeRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class.defaults().replace({
            'column': 3,
            'min': datetime(2013, 11, 8),
            'max': datetime(2013, 12, 3),
            'strict': True,
            'inclusive': True
        })
        filter_primitive = datetime_range_filter.DatetimeRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertGreater(new_dataframe['Date'].apply(parser.parse).min(), datetime(2013, 11, 8))
        self.assertLess(new_dataframe['Date'].apply(parser.parse).max(), datetime(2013, 12, 3))
        self.assertEqual(15, len(new_dataframe))

    def test_inclusive_permissive(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = datetime_range_filter.DatetimeRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class.defaults().replace({
            'column': 3,
            'min': datetime(2013, 11, 8),
            'max': datetime(2013, 12, 3),
            'strict': False,
            'inclusive': True
        })
        filter_primitive = datetime_range_filter.DatetimeRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertGreaterEqual(new_dataframe['Date'].apply(parser.parse).min(), datetime(2013, 11, 8))
        self.assertLessEqual(new_dataframe['Date'].apply(parser.parse).max(), datetime(2013, 12, 3))
        self.assertEqual(17, len(new_dataframe))

    def test_exclusive_strict(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = datetime_range_filter \
            .DatetimeRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class.defaults().replace({
            'column': 3,
            'min': datetime(2013, 11, 8),
            'max': datetime(2013, 12, 3),
            'strict': True,
            'inclusive': False
        })
        filter_primitive = datetime_range_filter.DatetimeRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertEqual(
            len(new_dataframe.loc[
                (new_dataframe['Date'].apply(parser.parse) >= datetime(2013, 11, 8)) &
                (new_dataframe['Date'].apply(parser.parse).max() <= datetime(2013, 12, 3))]), 0)
        self.assertEqual(23, len(new_dataframe))

    def test_exclusive_permissive(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = datetime_range_filter \
            .DatetimeRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class.defaults().replace({
            'column': 3,
            'min': datetime(2013, 11, 8),
            'max': datetime(2013, 12, 3),
            'strict': False,
            'inclusive': False
        })
        filter_primitive = datetime_range_filter.DatetimeRangeFilterPrimitive(hyperparams=hp)
        new_dataframe = filter_primitive.produce(inputs=resource).value

        self.assertEqual(
            len(new_dataframe.loc[
                (new_dataframe['Date'].apply(parser.parse) > datetime(2013, 11, 8)) &
                (new_dataframe['Date'].apply(parser.parse).max() < datetime(2013, 12, 3))]), 0)
        self.assertEqual(25, len(new_dataframe))

    def test_row_metadata_removal(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # add metadata for rows 0 and 1
        dataset.metadata = dataset.metadata.update(('learningData', 0), {'a': 0})
        dataset.metadata = dataset.metadata.update(('learningData', 5), {'b': 1})

        resource = test_utils.get_dataframe(dataset)

        # apply filter that removes rows 0 and 1
        filter_hyperparams_class = datetime_range_filter.DatetimeRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class.defaults().replace({
            'column': 3,
            'min': datetime(2013, 11, 4),
            'max': datetime(2013, 11, 7),
            'strict': True,
            'inclusive': False
        })
        filter_primitive = datetime_range_filter.DatetimeRangeFilterPrimitive(hyperparams=hp)
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

        filter_hyperparams_class = datetime_range_filter \
            .DatetimeRangeFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class.defaults().replace({
            'column': 1,
            'min': datetime(2013, 11, 1),
            'max': datetime(2013, 11, 4),
            'strict': False,
            'inclusive': False,
        })
        filter_primitive = datetime_range_filter.DatetimeRangeFilterPrimitive(hyperparams=hp)
        with self.assertRaises(ValueError):
            filter_primitive.produce(inputs=resource)


if __name__ == '__main__':
    unittest.main()
