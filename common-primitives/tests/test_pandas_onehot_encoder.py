import unittest
import pandas as pd

from d3m import container, utils
from common_primitives.pandas_onehot_encoder import PandasOneHotEncoderPrimitive
from d3m.metadata import base as metadata_base

import utils as test_utils


class PandasOneHotEncoderPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        training = pd.DataFrame({'Name': ['Henry', 'Diane', 'Kitty', 'Peter']})
        training = container.DataFrame(training, generate_metadata=True)
        training.metadata = training.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/CategoricalData',)
        training.metadata = training.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Attribute',)

        testing = pd.DataFrame({'Name': ['John', 'Alex','Henry','Diane']})
        testing = container.DataFrame(testing, generate_metadata=True)
        testing.metadata = testing.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        testing.metadata = testing.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Attribute',)
        testing.metadata = testing.metadata.update_column(0, {
            'custom_metadata': 42,
        })

        Hyperparams = PandasOneHotEncoderPrimitive.metadata.get_hyperparams()
        ht = PandasOneHotEncoderPrimitive(hyperparams=Hyperparams.defaults())

        ht.set_training_data(inputs=training)
        ht.fit()

        result_df = ht.produce(inputs=testing).value

        self.assertEqual(list(result_df.columns), ['Name_Diane', 'Name_Henry', 'Name_Kitty', 'Name_Peter'])

        self.assertEqual(list(result_df['Name_Henry']), [0, 0, 1, 0])
        self.assertEqual(list(result_df['Name_Diane']), [0, 0, 0, 1])
        self.assertEqual(list(result_df['Name_Kitty']), [0, 0, 0, 0])
        self.assertEqual(list(result_df['Name_Peter']), [0, 0, 0, 0])

        self.assertEqual(test_utils.convert_metadata(utils.to_json_structure(result_df.metadata.to_internal_simple_structure())), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Diane',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Henry',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Kitty',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Peter',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }])

        ht = PandasOneHotEncoderPrimitive(hyperparams=Hyperparams.defaults().replace({
            'dummy_na': True,
        }))

        ht.set_training_data(inputs=training)
        ht.fit()

        result_df = ht.produce(inputs=testing).value

        self.assertEqual(list(result_df.columns), ['Name_Diane', 'Name_Henry', 'Name_Kitty', 'Name_Peter', 'Name_nan'])

        self.assertEqual(list(result_df['Name_Henry']), [0, 0, 1, 0])
        self.assertEqual(list(result_df['Name_Diane']), [0, 0, 0, 1])
        self.assertEqual(list(result_df['Name_Kitty']), [0, 0, 0, 0])
        self.assertEqual(list(result_df['Name_Peter']), [0, 0, 0, 0])
        self.assertEqual(list(result_df['Name_nan']), [1, 1, 0, 0])

        self.assertEqual(test_utils.convert_metadata(utils.to_json_structure(result_df.metadata.to_internal_simple_structure())), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Diane',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Henry',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Kitty',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_Peter',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'custom_metadata': 42,
                'name': 'Name_nan',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }])


if __name__ == '__main__':
    unittest.main()
