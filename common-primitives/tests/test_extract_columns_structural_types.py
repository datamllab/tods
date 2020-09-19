import os
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_structural_types, column_parser

import utils as test_utils


class ExtractColumnsByStructuralTypesPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        hyperparams_class = extract_columns_structural_types.ExtractColumnsByStructuralTypesPrimitive.metadata.get_hyperparams()

        primitive = extract_columns_structural_types.ExtractColumnsByStructuralTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'structural_types': ('int',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

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
                'length': 2,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'int',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {
            'name': 'species',
            'structural_type': 'int',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })


if __name__ == '__main__':
    unittest.main()
