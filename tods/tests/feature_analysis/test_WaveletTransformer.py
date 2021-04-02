import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.feature_analysis.WaveletTransform import WaveletTransformPrimitive
import numpy as np
import pandas as pd
from d3m.container import DataFrame as d3m_dataframe
import os

class WaveletTransformerTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff=None
        curr_path = os.path.dirname(__file__)
        dataset_fname = os.path.join(curr_path, '../../../datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv')
        dataset = pd.read_csv(dataset_fname)
        # print(dataset.columns)
        value = dataset['value']
        main = d3m_dataframe(value, generate_metadata=True)

        ################## Test Wavelet transform ##################

        hyperparams_default = WaveletTransformPrimitive.metadata.get_hyperparams().defaults()
        hyperparams = hyperparams_default.replace({'wavelet': 'db8',
                                                   'level': 2,
                                                   'inverse': 0,
                                                   'return_result': 'new'})

        primitive = WaveletTransformPrimitive(hyperparams=hyperparams)
        new_main = primitive._produce(inputs=main).value

        # print(new_main)
        # print(mean_mse, std_mse)

        # self.assertAlmostEqual(mean_mse.__float__(), 0., delta=1e-8)
        # self.assertAlmostEquael(std_mse.__float__(), 0., delta=1e-8)

        # print(main.metadata.to_internal_simple_structure())
        # print(new_main.metadata.to_internal_simple_structure())

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
                    'length': 3521,
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
                'name': 'value',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'output_1',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'output_2',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }])

        ################## Test inverse transform ##################

        hyperparams = hyperparams_default.replace({'inverse': 1})

        primitive = WaveletTransformPrimitive(hyperparams=hyperparams)
        main_recover = primitive._produce(inputs=main).value

        self.assertAlmostEqual(main_recover.values.tolist(), main.values.tolist(), delta=1e-6)
        # print(main.metadata.to_internal_simple_structure())
        # print(main_recover.metadata.to_internal_simple_structure())

        self.assertEqual(utils.to_json_structure(main_recover.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 7027,
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
                'name': 'value',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }])


        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
