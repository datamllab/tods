from d3m import container,utils
from d3m.metadata import base as metadata_base
import unittest
from tods.feature_analysis import FastFourierTransform

import utils as test_utils
import  os
import numpy as np
import pandas as pd
import logging
from scipy.fft import fft
from cmath import polar

class FftTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff=None
        column_index =0


        main = container.DataFrame({'A': [1, 2, 3], 'B': ['a','b','c']},
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
                    'length': 3,
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

        hyperparams_class = FastFourierTransform.FastFourierTransformPrimitive.metadata.get_hyperparams()
        hp = hyperparams_class.defaults().replace({
            'use_semantic_types':True,
            'use_columns': (0,),
            'return_result':'append',
        })
        primitive = FastFourierTransform.FastFourierTransformPrimitive(hyperparams=hp)
        new_main = primitive._produce(inputs=main).value

        c = pd.DataFrame({"A":[1,2,3], "B":['a','b','c'],'A_fft_abs':[6.000000,1.732051,1.732051],'A_fft_phse':[-0.000000,2.617994,-2.617994]})

        pd.testing.assert_frame_equal(new_main, c)

        params = primitive.get_params()
        primitive.set_params(params=params)


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
                    'length': 4,
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
                'name': 'A_fft_abs',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
                }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'A_fft_phse',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }])


if __name__ == '__main__':
    unittest.main()
