import unittest
import os

from d3m import container

from common_primitives import audio_reader, cut_audio, dataset_to_dataframe, denormalize, column_parser


class AudioReaderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'audio_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        denormalize_hyperparams_class = denormalize.DenormalizePrimitive.metadata.get_hyperparams()
        denormalize_primitive = denormalize.DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        dataset = denormalize_primitive.produce(inputs=dataset).value

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        column_parser_hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        column_parser_primitive = column_parser.ColumnParserPrimitive(hyperparams=column_parser_hyperparams_class.defaults())
        dataframe = column_parser_primitive.produce(inputs=dataframe).value

        audio_hyperparams_class = audio_reader.AudioReaderPrimitive.metadata.get_hyperparams()
        audio_primitive = audio_reader.AudioReaderPrimitive(hyperparams=audio_hyperparams_class.defaults())
        dataframe = audio_primitive.produce(inputs=dataframe).value

        self.assertEqual(dataframe.iloc[0, 1], 'test_audio.mp3')
        self.assertEqual(dataframe.iloc[0, 5].shape, (4410, 1))

        cut_audio_hyperparams_class = cut_audio.CutAudioPrimitive.metadata.get_hyperparams()
        cut_audio_primitive = cut_audio.CutAudioPrimitive(hyperparams=cut_audio_hyperparams_class.defaults())
        dataframe = cut_audio_primitive.produce(inputs=dataframe).value

        self.assertEqual(dataframe.iloc[0, 1], 'test_audio.mp3')
        self.assertEqual(dataframe.iloc[0, 5].shape, (44, 1))

        self._test_metadata(dataframe.metadata, False)

    def _test_metadata(self, dataframe_metadata, is_can_accept):
        self.assertEqual(dataframe_metadata.query_column(2), {
            'structural_type': float,
            'name': 'start',
            'semantic_types': (
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Boundary',
                'https://metadata.datadrivendiscovery.org/types/IntervalStart',
            ),
        })
        self.assertEqual(dataframe_metadata.query_column(3), {
            'structural_type': float,
            'name': 'end',
            'semantic_types': (
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Boundary',
                'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
            ),
        })

        if is_can_accept:
            self.assertEqual(dataframe_metadata.query_column(5), {
                'structural_type': container.ndarray,
                'semantic_types': (
                    'http://schema.org/AudioObject',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    'https://metadata.datadrivendiscovery.org/types/UniqueKey',
                ),
                'name': 'filename',
            })
            self.assertEqual(dataframe_metadata.query((0, 5)), {
                'structural_type': container.ndarray,
                'semantic_types': (
                    'http://schema.org/AudioObject',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    'https://metadata.datadrivendiscovery.org/types/UniqueKey',
                ),
                'name': 'filename',
            })
        else:
            self.assertEqual(dataframe_metadata.query_column(5), {
                'structural_type': container.ndarray,
                'semantic_types': (
                    'http://schema.org/AudioObject',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    'https://metadata.datadrivendiscovery.org/types/UniqueKey',
                    'https://metadata.datadrivendiscovery.org/types/Table',
                ),
                'dimension': {
                    # The length is set here only because there is only one row.
                    'length': 44,
                    'name': 'rows',
                    'semantic_types': (
                        'https://metadata.datadrivendiscovery.org/types/TabularRow',
                    ),
                },
                'name': 'filename',
            })
            self.assertEqual(dataframe_metadata.query((0, 5)), {
                'structural_type': container.ndarray,
                'semantic_types': (
                    'http://schema.org/AudioObject',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                    'https://metadata.datadrivendiscovery.org/types/UniqueKey',
                    'https://metadata.datadrivendiscovery.org/types/Table',
                ),
                'dimension': {
                    'length': 44,
                    'name': 'rows',
                    'semantic_types': (
                        'https://metadata.datadrivendiscovery.org/types/TabularRow',
                    ),
                    'sampling_rate': 44100,
                },
                'name': 'filename',
            })


if __name__ == '__main__':
    unittest.main()
