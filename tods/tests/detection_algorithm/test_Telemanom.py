import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.detection_algorithm.Telemanom import TelemanomPrimitive


class TelemanomTest(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None
        main = container.DataFrame({'a': [1., 2., 3., 4.,5,6,7,8,9], 'b': [2., 3., 4., 5.,6,7,8,9,10], 'c': [3., 4., 5., 6.,7,8,9,10,11]},
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
                    'length': 9,
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

        hyperparams_class = TelemanomPrimitive.metadata.get_hyperparams()
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams.replace({'l_s': 2,'n_predictions':1,'return_result':'new','return_subseq_inds':True,'use_columns':(0,1,2)})

        # print("hyperparams",hyperparams)

        primitive = TelemanomPrimitive(hyperparams=hyperparams)
        primitive.set_training_data(inputs=main)
        primitive.fit()
        new_main = primitive.produce_score(inputs=main).value
        
        print("new main",new_main)

        # print(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()))
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
                    'length': 6,
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
            'metadata': {
                'name': 'Telemanom0_0',
                'structural_type': 'numpy.int64',     
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'Telemanom0_1',
                'structural_type': 'numpy.int64',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'Telemanom0_2',
                'structural_type': 'numpy.int64', 
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
            }
        }])


if __name__ == '__main__':
    unittest.main()





