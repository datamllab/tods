import unittest

from d3m import container

from common_primitives import unseen_label_encoder, unseen_label_decoder


class UnseenLabelEncoderTestCase(unittest.TestCase):
    def test_basic(self):
        encoder_hyperparams_class = unseen_label_encoder.UnseenLabelEncoderPrimitive.metadata.get_hyperparams()
        encoder_primitive = unseen_label_encoder.UnseenLabelEncoderPrimitive(hyperparams=encoder_hyperparams_class.defaults())

        inputs = container.DataFrame({
            'value': [0.0, 1.0, 2.0, 3.0],
            'number': [0, 1, 2, 3],
            'word': ['one', 'two', 'three', 'four'],
        }, generate_metadata=True)
        inputs.metadata = inputs.metadata.update_column(2, {
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData'],
        })

        encoder_primitive.set_training_data(inputs=inputs)
        encoder_primitive.fit()

        inputs = container.DataFrame({
            'value': [1.0, 2.0, 3.0],
            'number': [1, 2, 3],
            'word': ['one', 'two', 'five'],
        }, generate_metadata=True)
        inputs.metadata = inputs.metadata.update_column(2, {
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData'],
        })

        outputs = encoder_primitive.produce(inputs=inputs).value

        decoder_hyperparams_class = unseen_label_decoder.UnseenLabelDecoderPrimitive.metadata.get_hyperparams()
        decoder_primitive = unseen_label_decoder.UnseenLabelDecoderPrimitive(hyperparams=decoder_hyperparams_class.defaults().replace({'encoder': encoder_primitive}))

        decoded = decoder_primitive.produce(inputs=outputs).value

        self.assertEqual(decoded.values.tolist(), [
            [1, 1.0, 'one'],
            [2, 2.0, 'two'],
            [3, 3.0, ''],
        ])

        self.assertEqual(decoded.metadata.query_column(2)['structural_type'], str)


if __name__ == '__main__':
    unittest.main()
