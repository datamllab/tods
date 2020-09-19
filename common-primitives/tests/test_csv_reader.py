import unittest
import os

from d3m import container

from common_primitives import dataset_to_dataframe, csv_reader


class CSVReaderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_2', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        csv_hyperparams_class = csv_reader.CSVReaderPrimitive.metadata.get_hyperparams()
        csv_primitive = csv_reader.CSVReaderPrimitive(hyperparams=csv_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        tables = csv_primitive.produce(inputs=dataframe).value

        self.assertEqual(tables.shape, (5, 1))

        self._test_metadata(tables.metadata)

    def _test_metadata(self, metadata):
        self.assertEqual(metadata.query_column(0)['structural_type'], container.DataFrame)
        self.assertEqual(metadata.query_column(0)['semantic_types'], ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/Timeseries', 'https://metadata.datadrivendiscovery.org/types/Table'))

        self.assertEqual(metadata.query_column(0, at=(0, 0)), {
            'structural_type': str,
            'name': 'time',
            'semantic_types': (
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Time',
            )
        })
        self.assertEqual(metadata.query_column(1, at=(0, 0)), {
            'structural_type': str,
            'name': 'value',
            'semantic_types': (
               'http://schema.org/Float',
               'https://metadata.datadrivendiscovery.org/types/Attribute',
            )
        })


if __name__ == '__main__':
    unittest.main()
