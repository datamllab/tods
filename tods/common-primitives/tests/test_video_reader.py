import unittest
import os

from d3m import container

from common_primitives import dataset_to_dataframe, video_reader


class VideoReaderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'video_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        video_hyperparams_class = video_reader.VideoReaderPrimitive.metadata.get_hyperparams()
        video_primitive = video_reader.VideoReaderPrimitive(hyperparams=video_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        videos = video_primitive.produce(inputs=dataframe).value

        self.assertEqual(videos.shape, (2, 1))
        self.assertEqual(videos.iloc[0, 0].shape, (408, 240, 320, 3))
        self.assertEqual(videos.iloc[1, 0].shape, (79, 240, 320, 3))

        self._test_metadata(videos.metadata)

    def _test_metadata(self, metadata):
        self.assertEqual(metadata.query_column(0)['structural_type'], container.ndarray)
        self.assertEqual(metadata.query_column(0)['semantic_types'], ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/VideoObject'))


if __name__ == '__main__':
    unittest.main()
