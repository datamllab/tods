import unittest
import os

from d3m import container

from common_primitives import audio_reader, dataset_to_dataframe, denormalize


class AudioReaderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'audio_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        audio_hyperparams_class = audio_reader.AudioReaderPrimitive.metadata.get_hyperparams()
        audio_primitive = audio_reader.AudioReaderPrimitive(hyperparams=audio_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        audios = audio_primitive.produce(inputs=dataframe).value

        self.assertEqual(audios.shape, (1, 1))
        self.assertEqual(audios.iloc[0, 0].shape, (4410, 1))

        self._test_metadata(audios.metadata, True)

        self.assertEqual(audios.metadata.query((0, 0))['dimension']['length'], 4410)
        self.assertEqual(audios.metadata.query((0, 0))['dimension']['sampling_rate'], 44100)

    def _test_metadata(self, metadata, is_table):
        semantic_types = ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'http://schema.org/AudioObject')

        if is_table:
            semantic_types += ('https://metadata.datadrivendiscovery.org/types/Table',)

        self.assertEqual(metadata.query_column(0)['name'], 'filename')
        self.assertEqual(metadata.query_column(0)['structural_type'], container.ndarray)
        self.assertEqual(metadata.query_column(0)['semantic_types'], semantic_types)

    def test_boundaries_reassign(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'audio_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        denormalize_hyperparams_class = denormalize.DenormalizePrimitive.metadata.get_hyperparams()
        denormalize_primitive = denormalize.DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        dataset = denormalize_primitive.produce(inputs=dataset).value

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        audio_hyperparams_class = audio_reader.AudioReaderPrimitive.metadata.get_hyperparams()
        audio_primitive = audio_reader.AudioReaderPrimitive(hyperparams=audio_hyperparams_class.defaults().replace({'return_result': 'append'}))
        audios = audio_primitive.produce(inputs=dataframe).value

        self.assertEqual(audios.shape, (1, 6))
        self.assertEqual(audios.iloc[0, 5].shape, (4410, 1))

        self._test_boundaries_reassign_metadata(audios.metadata, True)

        self.assertEqual(audios.metadata.query((0, 5))['dimension']['length'], 4410)
        self.assertEqual(audios.metadata.query((0, 5))['dimension']['sampling_rate'], 44100)

    def _test_boundaries_reassign_metadata(self, metadata, is_table):
        semantic_types = ('http://schema.org/AudioObject', 'https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/UniqueKey')

        if is_table:
            semantic_types += ('https://metadata.datadrivendiscovery.org/types/Table',)

        self.assertEqual(metadata.query_column(5)['name'], 'filename')
        self.assertEqual(metadata.query_column(5)['structural_type'], container.ndarray)
        self.assertEqual(metadata.query_column(5)['semantic_types'], semantic_types)

        self.assertEqual(metadata.query_column(2), {
            'structural_type': str,
            'name': 'start',
            'semantic_types': (
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Boundary',
                'https://metadata.datadrivendiscovery.org/types/IntervalStart',
            ),
            'boundary_for': {
                'resource_id': 'learningData',
                'column_index': 5,
            },
        })
        self.assertEqual(metadata.query_column(3), {
            'structural_type': str,
            'name': 'end',
            'semantic_types': (
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Boundary',
                'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
            ),
            'boundary_for': {
                'resource_id': 'learningData',
                'column_index': 5,
            },
        })


if __name__ == '__main__':
    unittest.main()
