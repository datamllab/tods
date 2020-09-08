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

from common_primitives import dataset_to_dataframe, csv_reader, denormalize, column_map, column_parser

import utils as test_utils


class ColumnMapTestCase(unittest.TestCase):
    def test_transformer(self):
        self.maxDiff = None

        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_2', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams = denormalize.DenormalizePrimitive.metadata.get_hyperparams()
        primitive = denormalize.DenormalizePrimitive(hyperparams=hyperparams.defaults())
        dataset = primitive.produce(inputs=dataset).value

        hyperparams = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams.defaults())
        dataframe = primitive.produce(inputs=dataset).value

        hyperparams = csv_reader.CSVReaderPrimitive.metadata.get_hyperparams()
        primitive = csv_reader.CSVReaderPrimitive(hyperparams=hyperparams.defaults().replace({'return_result': 'replace'}))
        dataframe = primitive.produce(inputs=dataframe).value

        hyperparams = column_map.DataFrameColumnMapPrimitive.metadata.get_hyperparams()
        primitive = column_map.DataFrameColumnMapPrimitive(
            # We have to make an instance of the primitive ourselves.
            hyperparams=hyperparams.defaults().replace({
                # First we use identity primitive which should not really change anything.
                'primitive': NullTransformerPrimitive(
                    hyperparams=NullTransformerPrimitive.metadata.get_hyperparams().defaults(),
                ),
            }),
        )
        mapped_dataframe = primitive.produce(inputs=dataframe).value

        self.assertEqual(test_utils.convert_through_json(test_utils.effective_metadata(dataframe.metadata)), test_utils.convert_through_json(test_utils.effective_metadata(mapped_dataframe.metadata)))

        self.assertEqual(test_utils.convert_through_json(dataframe), test_utils.convert_through_json(mapped_dataframe))

        primitive = column_map.DataFrameColumnMapPrimitive(
            # We have to make an instance of the primitive ourselves.
            hyperparams=hyperparams.defaults().replace({
                'primitive': column_parser.ColumnParserPrimitive(
                    hyperparams=column_parser.ColumnParserPrimitive.metadata.get_hyperparams().defaults(),
                ),
            }),
        )
        dataframe = primitive.produce(inputs=mapped_dataframe).value

        self.assertEqual(test_utils.convert_through_json(dataframe)[0][1][0], [0, 2.6173])

        pickle.dumps(primitive)


if __name__ == '__main__':
    unittest.main()
