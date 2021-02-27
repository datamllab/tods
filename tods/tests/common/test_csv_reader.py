import unittest
import os

from d3m import container

from tods.common import CSVReader 
from tods.data_processing import DatasetToDataframe 


class CSVReaderPrimitiveTestCase(unittest.TestCase):

    def _get_yahoo_dataset(self):    # pragma: no cover
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        return dataset
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        csv_hyperparams_class = CSVReader.CSVReaderPrimitive.metadata.get_hyperparams()
        csv_primitive = CSVReader.CSVReaderPrimitive(hyperparams=csv_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        tables = csv_primitive.produce(inputs=dataframe).value

        self.assertEqual(tables.shape, (1260, 8))


if __name__ == '__main__':
    unittest.main()
