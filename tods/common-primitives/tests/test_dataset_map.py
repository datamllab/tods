import unittest
import os
import pickle
import sys

from d3m import container, index, utils as d3m_utils

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')
sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.null import NullTransformerPrimitive, NullUnsupervisedLearnerPrimitive

# To hide any logging or stdout output.
with d3m_utils.silence():
    index.register_primitive('d3m.primitives.operator.null.TransformerTest', NullTransformerPrimitive)
    index.register_primitive('d3m.primitives.operator.null.UnsupervisedLearnerTest', NullUnsupervisedLearnerPrimitive)

from common_primitives import dataset_to_dataframe, denormalize, dataset_map, column_parser

import utils as test_utils


class DatasetMapTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None

        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # First we try denormalizing and column parsing.
        hyperparams = denormalize.DenormalizePrimitive.metadata.get_hyperparams()
        primitive = denormalize.DenormalizePrimitive(hyperparams=hyperparams.defaults())
        dataset_1 = primitive.produce(inputs=dataset).value

        hyperparams = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams.defaults())
        dataframe_1 = primitive.produce(inputs=dataset_1).value

        hyperparams = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams.defaults().replace({'return_result': 'replace'}))
        dataframe_1 = primitive.produce(inputs=dataframe_1).value

        # Second we try first column parsing and then denormalizing.
        hyperparams = dataset_map.DataFrameDatasetMapPrimitive.metadata.get_hyperparams()
        primitive = dataset_map.DataFrameDatasetMapPrimitive(
            # We have to make an instance of the primitive ourselves.
            hyperparams=hyperparams.defaults().replace({
                'primitive': column_parser.ColumnParserPrimitive(
                    hyperparams=column_parser.ColumnParserPrimitive.metadata.get_hyperparams().defaults(),
                ),
                'resources': 'all',
            }),

        )
        dataset_2 = primitive.produce(inputs=dataset).value

        hyperparams = denormalize.DenormalizePrimitive.metadata.get_hyperparams()
        primitive = denormalize.DenormalizePrimitive(hyperparams=hyperparams.defaults())
        dataset_2 = primitive.produce(inputs=dataset_2).value

        hyperparams = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams.defaults())
        dataframe_2 = primitive.produce(inputs=dataset_2).value

        self.assertEqual(test_utils.convert_through_json(dataframe_1), test_utils.convert_through_json(dataframe_2))
        self.assertEqual(dataframe_1.metadata.to_internal_json_structure(), dataframe_2.metadata.to_internal_json_structure())

        pickle.dumps(primitive)


if __name__ == '__main__':
    unittest.main()
