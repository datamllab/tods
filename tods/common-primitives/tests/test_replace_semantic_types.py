import os
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, replace_semantic_types

import utils as test_utils


class ReplaceSemanticTypesPrimitiveTestCase(unittest.TestCase):
    def _get_iris_dataframe(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        return dataframe

    def test_basic(self):
        dataframe = self._get_iris_dataframe()

        hyperparams_class = replace_semantic_types.ReplaceSemanticTypesPrimitive.metadata.get_hyperparams()
        primitive = replace_semantic_types.ReplaceSemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'from_semantic_types': ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',),
            'to_semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',),
        }))

        outputs = primitive.produce(inputs=dataframe).value

        self._test_metadata(outputs.metadata)

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
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

        self.assertTrue(metadata.get_elements((metadata_base.ALL_ELEMENTS,)) in [[0, 1, 2, 3, 4, 5], [metadata_base.ALL_ELEMENTS, 0, 1, 2, 3, 4, 5]])


if __name__ == '__main__':
    unittest.main()
