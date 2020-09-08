import math
import os.path
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, grouping_field_compose

import utils as test_utils


class GroupingFieldComposePrimitiveTestCase(unittest.TestCase):
    def test_compose_two_suggested_fields(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_3', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        compose_hyperparams_class = grouping_field_compose.GroupingFieldComposePrimitive.metadata.get_hyperparams()
        hp = compose_hyperparams_class.defaults().replace({
            'join_char': '-',
            'output_name': 'grouping'
        })
        compose_primitive = grouping_field_compose.GroupingFieldComposePrimitive(hyperparams=hp)
        new_dataframe = compose_primitive.produce(inputs=resource).value

        self.assertEqual(new_dataframe.shape, (40, 6))
        self.assertEqual('abbv-2013', new_dataframe['grouping'][0])

        col_meta = new_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 5))
        self.assertEqual(col_meta['name'], 'grouping')
        self.assertTrue('https://metadata.datadrivendiscovery.org/types/GroupingKey' in col_meta['semantic_types'])

    def test_compose_two_specified_fields(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_3', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        compose_hyperparams_class = grouping_field_compose.GroupingFieldComposePrimitive.metadata.get_hyperparams()
        hp = compose_hyperparams_class.defaults().replace({
            'columns': [1,3],
            'join_char': '-',
            'output_name': 'grouping'
        })
        compose_primitive = grouping_field_compose.GroupingFieldComposePrimitive(hyperparams=hp)
        new_dataframe = compose_primitive.produce(inputs=resource).value

        self.assertEqual(new_dataframe.shape, (40, 6))
        self.assertEqual('abbv-11-01', new_dataframe['grouping'][0])

        col_meta = new_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 5))
        self.assertEqual(col_meta['name'], 'grouping')
        self.assertTrue('https://metadata.datadrivendiscovery.org/types/GroupingKey' in col_meta['semantic_types'])

if __name__ == '__main__':
    unittest.main()
