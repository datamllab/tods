import unittest
import os

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, csv_reader, dataframe_flatten


class DataFrameFlattenPrimitiveTestCase(unittest.TestCase):

    COLUMN_METADATA = {
        'time': {
            'structural_type': str,
            'name': 'time',
            'semantic_types': (
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Time'
            ),
        },
        'value': {
            'structural_type': str,
            'name': 'value',
            'semantic_types': (
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute'
            ),
        }
    }

    def test_replace(self) -> None:
        tables = self._load_data()
        flat_hyperparams_class = dataframe_flatten.DataFrameFlattenPrimitive.metadata.get_hyperparams()
        flat_primitive = dataframe_flatten.DataFrameFlattenPrimitive(hyperparams=flat_hyperparams_class.defaults())
        flat_result = flat_primitive.produce(inputs=tables).value

        self.assertEqual(flat_result.shape, (830, 3))

        metadata = flat_result.metadata
        self._check_filename_metadata(metadata, 0)
        self.assertEqual(metadata.query_column(1), self.COLUMN_METADATA['time'])
        self.assertEqual(metadata.query_column(2), self.COLUMN_METADATA['value'])

    def test_new(self) -> None:
        tables = self._load_data()

        flat_hyperparams_class = dataframe_flatten.DataFrameFlattenPrimitive.metadata.get_hyperparams()
        hp = flat_hyperparams_class.defaults().replace({
            'return_result': 'new',
            'add_index_columns': False
        })
        flat_primitive = dataframe_flatten.DataFrameFlattenPrimitive(hyperparams=hp)
        flat_result = flat_primitive.produce(inputs=tables).value

        self.assertEqual(flat_result.shape, (830, 2))
        metadata = flat_result.metadata
        self.assertEqual(metadata.query_column(0), self.COLUMN_METADATA['time'])
        self.assertEqual(metadata.query_column(1), self.COLUMN_METADATA['value'])

    def test_add_index_columns(self) -> None:
        tables = self._load_data()

        flat_hyperparams_class = dataframe_flatten.DataFrameFlattenPrimitive.metadata.get_hyperparams()
        hp = flat_hyperparams_class.defaults().replace({
            'return_result': 'new',
            'add_index_columns': True
        })
        flat_primitive = dataframe_flatten.DataFrameFlattenPrimitive(hyperparams=hp)
        flat_result = flat_primitive.produce(inputs=tables).value

        self.assertEqual(flat_result.shape, (830, 3))
        metadata = flat_result.metadata
        self._check_filename_metadata(metadata, 0)
        self.assertEqual(metadata.query_column(1), self.COLUMN_METADATA['time'])
        self.assertEqual(metadata.query_column(2), self.COLUMN_METADATA['value'])

    def test_use_columns(self) -> None:
        tables = self._load_data()

        flat_hyperparams_class = dataframe_flatten.DataFrameFlattenPrimitive.metadata.get_hyperparams()
        hp = flat_hyperparams_class.defaults().replace({'use_columns': [1]})

        flat_primitive = dataframe_flatten.DataFrameFlattenPrimitive(hyperparams=hp)
        flat_result = flat_primitive.produce(inputs=tables).value

        self.assertEqual(flat_result.shape, (830, 3), [0])

        metadata = flat_result.metadata
        self._check_filename_metadata(metadata, 0)
        self.assertEqual(metadata.query_column(1), self.COLUMN_METADATA['time'])
        self.assertEqual(metadata.query_column(2), self.COLUMN_METADATA['value'])

    def test_exclude_columns(self) -> None:
        tables = self._load_data()

        flat_hyperparams_class = dataframe_flatten.DataFrameFlattenPrimitive.metadata.get_hyperparams()
        hp = flat_hyperparams_class.defaults().replace({'exclude_columns': [0]})

        flat_primitive = dataframe_flatten.DataFrameFlattenPrimitive(hyperparams=hp)
        flat_result = flat_primitive.produce(inputs=tables).value

        self.assertEqual(flat_result.shape, (830, 3), [0])

        metadata = flat_result.metadata
        self._check_filename_metadata(metadata, 0)
        self.assertEqual(metadata.query_column(1), self.COLUMN_METADATA['time'])
        self.assertEqual(metadata.query_column(2), self.COLUMN_METADATA['value'])

    def _load_data(self) -> container.DataFrame:
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_2', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        csv_hyperparams_class = csv_reader.CSVReaderPrimitive.metadata.get_hyperparams()
        csv_primitive = csv_reader.CSVReaderPrimitive(hyperparams=csv_hyperparams_class.defaults().replace({'return_result': 'append'}))
        return csv_primitive.produce(inputs=dataframe).value

    def _check_filename_metadata(self, metadata: metadata_base.Metadata, col_num: int) -> None:
        self.assertEqual(metadata.query_column(col_num)['name'], 'filename')
        self.assertEqual(metadata.query_column(col_num)['structural_type'], str)
        self.assertEqual(metadata.query_column(col_num)['semantic_types'], (
            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            'https://metadata.datadrivendiscovery.org/types/FileName',
            'https://metadata.datadrivendiscovery.org/types/Timeseries'))


if __name__ == '__main__':
    unittest.main()
