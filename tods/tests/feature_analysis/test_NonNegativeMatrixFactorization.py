from d3m import container
from d3m.metadata import base as metadata_base
import unittest
from tods.feature_analysis import NonNegativeMatrixFactorization
from d3m import container,utils
from d3m.container import DataFrame as d3m_dataframe

import utils as test_utils
import  os
import numpy as np
import pandas as pd
import logging
from scipy.fft import fft
from cmath import polar
import nimfa

LENGTH = 1400

class NmfTestCase(unittest.TestCase):
	def test_basic(self):
		self.maxDiff=None

		main = container.DataFrame({'A': [1, 2, 3], 'B': [4,5,6]},
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
				'structural_type': 'numpy.int64', 
				'name': 'B',
			},
		}])

		a = np.array([[1,0,1,0,1],[1,0,1,0,1],[1,0,1,0,1]])
		b = np.array([[1,0],[1,0],[1,0],[1,0],[1,0]])

		hyperparams_class = NonNegativeMatrixFactorization.NonNegativeMatrixFactorizationPrimitive.metadata.get_hyperparams()
		hp = hyperparams_class.defaults().replace({
			'use_semantic_types': True,
		    'use_columns': (0,1,),
		    'return_result':'append',
		    'rank':5,
		    'seed':'fixed',
		    'W':a,
		    'H': b,
		})
		primitive = NonNegativeMatrixFactorization.NonNegativeMatrixFactorizationPrimitive(hyperparams=hp)
		new_main = primitive._produce(inputs=main).value

		print("new_main",new_main)
		c = pd.DataFrame({"A":[1,2,3,np.nan,np.nan], "B":[4,5,6,np.nan,np.nan],
                                  'row_latent_vector_0':[0.816725,1.078965,1.341205,np.nan,np.nan],
                                  'row_latent_vector_1':[3.514284e-16,2.383547e-16,2.227207e-16,np.nan,np.nan],
                                  'row_latent_vector_2':[0.816725,1.078965,1.341205,np.nan,np.nan],
                                  'row_latent_vector_3':[3.514284e-16,2.383547e-16,2.227207e-16,np.nan,np.nan],
                                  'row_latent_vector_4':[0.816725,1.078965,1.341205,np.nan,np.nan],
                                  'column_latent_vector_0':[ 0.642626,0.542312,0.642626,0.542312,0.642626],
                                  'column_latent_vector_1':[ 1.534324,1.848782,1.534324,1.848782,1.534324],
                                  })
		# pd.testing.assert_frame_equal(new_main, c)

		params = primitive.get_params()
		primitive.set_params(params=params)


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
					'length': 3,
				},
			},
		}, {
			'selector': ['__ALL_ELEMENTS__'],
			'metadata': {
				'dimension': {
					'name': 'columns',
					'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
					'length': 9,
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
				'structural_type': 'numpy.int64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 2],
			'metadata': {
				'name': 'row_latent_vector_0',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 3],
			'metadata': {
				'name': 'row_latent_vector_1',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 4],
			'metadata': {
				'name': 'row_latent_vector_2',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 5],
			'metadata': {
				'name': 'row_latent_vector_3',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 6],
			'metadata': {
				'name': 'row_latent_vector_4',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 7],
			'metadata': {
				'name': 'column_latent_vector_0',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}, {
			'selector': ['__ALL_ELEMENTS__', 8],
			'metadata': {
				'name': 'column_latent_vector_1',
				'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
				'structural_type': 'numpy.float64',
			},
		}])


		hyperparams_class = NonNegativeMatrixFactorization.NonNegativeMatrixFactorizationPrimitive.metadata.get_hyperparams()
		hp = hyperparams_class.defaults().replace({
			'use_semantic_types': False,
		    'use_columns': (0,1,),
		    'return_result':'append',
		    'rank':5,
		    'seed':'fixed',
		    'W':a,
		    'H': b,
		})
		primitive = NonNegativeMatrixFactorization.NonNegativeMatrixFactorizationPrimitive(hyperparams=hp)
		new_main = primitive.produce(inputs=main).value


		params = primitive.get_params()
		primitive.set_params(params=params)
		


if __name__ == '__main__':
	unittest.main()
