import unittest
import os

from d3m import container

from common_primitives import dataset_to_dataframe, dataframe_image_reader


class DataFrameImageReaderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'image_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        image_hyperparams_class = dataframe_image_reader.DataFrameImageReaderPrimitive.metadata.get_hyperparams()
        image_primitive = dataframe_image_reader.DataFrameImageReaderPrimitive(hyperparams=image_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        images = image_primitive.produce(inputs=dataframe).value

        self.assertEqual(images.shape, (5, 1))
        self.assertEqual(images.iloc[0, 0].shape, (225, 150, 3))
        self.assertEqual(images.iloc[1, 0].shape, (32, 32, 3))
        self.assertEqual(images.iloc[2, 0].shape, (32, 32, 3))
        self.assertEqual(images.iloc[3, 0].shape, (28, 28, 1))
        self.assertEqual(images.iloc[4, 0].shape, (28, 28, 1))

        self._test_metadata(images.metadata)

        self.assertEqual(images.metadata.query((0, 0))['image_reader_metadata'], {
            'jfif': 257,
            'jfif_version': (1, 1),
            'dpi': (96, 96),
            'jfif_unit': 1,
            'jfif_density': (96, 96),
        })

    def _test_metadata(self, metadata):
        self.assertEqual(metadata.query_column(0)['structural_type'], container.ndarray)
        self.assertEqual(metadata.query_column(0)['semantic_types'], ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/ImageObject'))


if __name__ == '__main__':
    unittest.main()
