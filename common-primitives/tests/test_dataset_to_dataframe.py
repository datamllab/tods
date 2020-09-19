import os
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe

import utils as test_utils


class DatasetToDataFramePrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        self.assertIsInstance(dataframe, container.DataFrame)

        for row in dataframe:
            for cell in row:
                # Nothing should be parsed from a string.
                self.assertIsInstance(cell, str)

        self.assertEqual(len(dataframe), 150)
        self.assertEqual(len(dataframe.iloc[0]), 6)

        self._test_metadata(dataframe.metadata)

    def _test_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 150,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 6,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        for i in range(1, 5):
            self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, i))), {
                'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'][i - 1],
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            }, i)

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))), {
            'name': 'species',
            'structural_type': 'str',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })


if __name__ == '__main__':
    unittest.main()
