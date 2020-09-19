import unittest
import os

from d3m import container

from common_primitives import dataset_to_dataframe, text_reader


class TextReaderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'text_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        text_hyperparams_class = text_reader.TextReaderPrimitive.metadata.get_hyperparams()
        text_primitive = text_reader.TextReaderPrimitive(hyperparams=text_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        tables = text_primitive.produce(inputs=dataframe).value

        self.assertEqual(tables.shape, (4, 1))

        self.assertEqual(tables.metadata.query_column(0)['structural_type'], str)
        self.assertEqual(tables.metadata.query_column(0)['semantic_types'], ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/Text'))


if __name__ == '__main__':
    unittest.main()
