import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.feature_analysis import SKTruncatedSVD


class SKTruncatedSVDTest(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None
        main = container.DataFrame({'a': [1., 2., 3.], 'b': [2., 3., 4.], 'c': [3., 4., 5.],},
                                    columns=['a', 'b', 'c'],
                                    generate_metadata=True)

        print(main)

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


        hyperparams_class = SKTruncatedSVD.SKTruncatedSVDPrimitive.metadata.get_hyperparams()
        primitive = SKTruncatedSVD.SKTruncatedSVDPrimitive(hyperparams=hyperparams_class.defaults())
        primitive.set_training_data(inputs=main)
        primitive.fit()
        new_main = primitive.produce(inputs=main).value
        print(new_main)

        # expected_output = container.DataFrame({'timestamp': [1., 4.],'a': [1., 3.], 'b': [2., 4.],})
        # self.assertEqual(new_main.values.tolist() , expected_output.values.tolist())        


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
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'a',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'b',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'c',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'Truncated SVD0_0',
                'structural_type': 'numpy.float64',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'Truncated SVD0_1',
                'structural_type': 'numpy.float64',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }])





if __name__ == '__main__':
    unittest.main()
