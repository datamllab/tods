import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.detection_algorithm.PyodCOF import COFPrimitive
import utils as test_utils
import pandas as pd

class COFTest(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None
        main = container.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.], 'c': [3., 4., 11.],},
                                    columns=['a', 'b', 'c'],
                                    generate_metadata=True)

        # print(main)


        self.assertEqual(utils.to_json_structure(main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'c'}
        }])


        self.assertIsInstance(main, container.DataFrame)


        hyperparams_class = COFPrimitive.metadata.get_hyperparams()
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams.replace({'return_result': 'new',

                                        })

        primitive = COFPrimitive(hyperparams=hyperparams)
        primitive.set_training_data(inputs=main)
        primitive.fit()
        new_main = primitive.produce(inputs=main).value
        nme2 = primitive.produce_score(inputs=main).value
        # print(type(new_main))

        c = pd.DataFrame({0:[0,0,1]})

        pd.testing.assert_frame_equal(new_main, c)

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'Connectivity-Based Outlier Factor (COF)0_0',
                'structural_type': 'numpy.int64',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'] 
            },
        }])


if __name__ == '__main__':
    unittest.main()
