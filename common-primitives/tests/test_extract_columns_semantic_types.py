import os
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_semantic_types

import utils as test_utils


class ExtractColumnsBySemanticTypePrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')}))

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
                'length': 5,
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

        self.assertTrue(metadata.get_elements((metadata_base.ALL_ELEMENTS,)) in [[0, 1, 2, 3, 4], [metadata_base.ALL_ELEMENTS, 0, 1, 2, 3, 4]])

    def test_set(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "datasets",
                "boston_dataset_1",
                "datasetDoc.json",
            )
        )

        dataset = container.Dataset.load(
            "file://{dataset_doc_path}".format(dataset_doc_path=dataset_doc_path)
        )

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 14),
            "https://metadata.datadrivendiscovery.org/types/Target",
        )
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 14),
            "https://metadata.datadrivendiscovery.org/types/TrueTarget",
        )
        dataset.metadata = dataset.metadata.remove_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 14),
            "https://metadata.datadrivendiscovery.org/types/Attribute",
        )

        hyperparams_class = (
            dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        )

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=hyperparams_class.defaults()
        )

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = (
            extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
        )

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {
                    "semantic_types": (
                        "https://metadata.datadrivendiscovery.org/types/Attribute",
                        "http://schema.org/Integer",
                    ),
                    "match_logic": "equal",
                }
            )
        )

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        self._test_equal_metadata(dataframe.metadata)

    def _test_equal_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(
            test_utils.convert_through_json(metadata.query(())),
            {
                "structural_type": "d3m.container.pandas.DataFrame",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/Table"
                ],
                "dimension": {
                    "name": "rows",
                    "semantic_types": [
                        "https://metadata.datadrivendiscovery.org/types/TabularRow"
                    ],
                    "length": 506,
                },
                "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/container.json",
            },
        )

        # only one column that should match
        self.assertEqual(
            test_utils.convert_through_json(
                metadata.query((metadata_base.ALL_ELEMENTS,))
            ),
            {
                "dimension": {
                    "name": "columns",
                    "semantic_types": [
                        "https://metadata.datadrivendiscovery.org/types/TabularColumn"
                    ],
                    "length": 1,
                }
            },
        )

        self.assertEqual(
            test_utils.convert_through_json(
                metadata.query((metadata_base.ALL_ELEMENTS, 0))
            ),
            {
                "name": "TAX",
                "structural_type": "str",
                "semantic_types": [
                    "http://schema.org/Integer",
                    "https://metadata.datadrivendiscovery.org/types/Attribute",
                ],
                "description": "full-value property-tax rate per $10,000",
            },
        )


if __name__ == '__main__':
    unittest.main()
