import math
import os.path
import unittest

from datetime import datetime
from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, datetime_field_compose

import utils as test_utils


class DatetimeFieldComposePrimitiveTestCase(unittest.TestCase):
    def test_compose_two_fields(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_3', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        compose_hyperparams_class = datetime_field_compose.DatetimeFieldComposePrimitive.metadata.get_hyperparams()
        hp = compose_hyperparams_class({
            'columns': [2,3],
            'join_char': '-',
            'output_name': 'timestamp'
        })
        compose_primitive = datetime_field_compose.DatetimeFieldComposePrimitive(hyperparams=hp)
        new_dataframe = compose_primitive.produce(inputs=resource).value

        self.assertEqual(new_dataframe.shape, (40, 6))
        self.assertEqual(datetime(2013, 11, 1), new_dataframe['timestamp'][0])

        col_meta = new_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 5))
        self.assertEqual(col_meta['name'], 'timestamp')
        self.assertTrue('https://metadata.datadrivendiscovery.org/types/Time' in col_meta['semantic_types'])

    def test_bad_join_char(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_3', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        compose_hyperparams_class = datetime_field_compose.DatetimeFieldComposePrimitive.metadata.get_hyperparams()
        hp = compose_hyperparams_class({
            'columns': [2,3],
            'join_char': 'cc',
            'output_name': 'timestamp'
        })
        compose_primitive = datetime_field_compose.DatetimeFieldComposePrimitive(hyperparams=hp)
        with self.assertRaises(ValueError):
            compose_primitive.produce(inputs=resource)

    def test_bad_columns(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_3', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        compose_hyperparams_class = datetime_field_compose.DatetimeFieldComposePrimitive.metadata.get_hyperparams()
        hp = compose_hyperparams_class({
            'columns': [1,2],
            'join_char': '-',
            'output_name': 'timestamp'
        })
        compose_primitive = datetime_field_compose.DatetimeFieldComposePrimitive(hyperparams=hp)
        with self.assertRaises(ValueError):
            compose_primitive.produce(inputs=resource)

if __name__ == '__main__':
    unittest.main()
