import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.data_processing import CategoricalToBinary
import numpy as np
import pandas as pd
import utils as test_utils
import os

class CategoricalBinaryTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff=None

        main = container.DataFrame({'A': [1, 2], 'B': ['a','b']},
                                    columns=['A', 'B'],
                                    generate_metadata=True)

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
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'structural_type': 'numpy.int64', 
                'name': 'A',
                },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'structural_type': 'str', 
                'name': 'B',
            },
        }])


        self.assertIsInstance(main, container.DataFrame)

        hyperparams_class = CategoricalToBinary.CategoricalToBinaryPrimitive.metadata.get_hyperparams()
        hp = hyperparams_class.defaults().replace({
            'use_semantic_types':True,
            'use_columns': (0,),
            'return_result':'append',
        })

        primitive = CategoricalToBinary.CategoricalToBinaryPrimitive(hyperparams=hp)
        new_main = primitive.produce(inputs=main).value

        c = pd.DataFrame({"A":[1,2], "B":['a','b'],"A_1.0":[np.uint8(1),np.uint8(0)],"A_2.0":[np.uint8(0),np.uint8(1)],"A_nan":[np.uint8(0),np.uint8(0)]})


        # print("new_main\n",new_main)
        # pd.testing.assert_frame_equal(new_main, c)
        

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
                    'length': 2,
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
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'str',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'A_1.0',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'A_2.0',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
                },
        },{
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'A_nan',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.uint8',
                },
        }])


        # print(new_main)
        # print(test_utils.convert_through_json(new_main.metadata.query(())))
        # print(test_utils.convert_through_json(new_main.metadata.query((metadata_base.ALL_ELEMENTS,))))
        # print(mean_mse, std_mse)

       
        # print("after testing")

        # self.assertAlmostEqual(mean_mse.__float__(), 0., delta=1e-8)
        # self.assertAlmostEqual(std_mse.__float__(), 0., delta=1e-8)

        # print(main.metadata.to_internal_simple_structure())
        # print(new_main.metadata.to_internal_simple_structure())

        params = primitive.get_params()
        primitive.set_params(params=params)



        hyperparams_class = CategoricalToBinary.CategoricalToBinaryPrimitive.metadata.get_hyperparams()
        hp = hyperparams_class.defaults().replace({
            'use_semantic_types':False,
            'use_columns': (0,),
            'return_result':'append',
        })

        primitive = CategoricalToBinary.CategoricalToBinaryPrimitive(hyperparams=hp)
        new_main = primitive.produce(inputs=main).value

        print("new_main \n",new_main)



if __name__ == '__main__':
    unittest.main()
