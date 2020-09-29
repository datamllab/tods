import unittest


from datetime import datetime

from d3m import container, utils
from d3m.metadata import base as metadata_base

from feature_analysis import AutoCorrelation


#import utils as test_utils

import numpy as np
import pandas as pd

class AutoCorrelationTestCase(unittest.TestCase):
	def test_basic(self):
		self.maxDiff = None
		main = container.DataFrame({'d3mIndex': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
									'timestamp': [1472918400, 1472918700, 1472919000, 1472919300,
													1472919600, 1472919900, 1472920200, 1472920500, 1472920800, 1472921100],
									'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 
									'ground_truth': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]},
			    columns = ['d3mIndex', 'timestamp', 'value', 'ground_truth'], generate_metadata = True)
		"""
		main.metadata = main.metadata.update_column(0, {'name': 'd3mIndex_'})
		main.metadata = main.metadata.update_column(1, {'name': 'timestamp_'})
		main.metadata = main.metadata.update_column(2, {'name': 'value_'})
		main.metadata = main.metadata.update_column(3, {'name': 'ground_truth_'})
		"""

		#print(main)

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
					'length': 10,
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
			'metadata': {'structural_type': 'numpy.int64', 'name': 'd3mIndex'},
		}, {
			'selector': ['__ALL_ELEMENTS__', 1],
			'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
		}, {
			'selector': ['__ALL_ELEMENTS__', 2],
			'metadata': {'structural_type': 'numpy.float64', 'name': 'value'},
		}, {
			'selector': ['__ALL_ELEMENTS__', 3],
			'metadata': {'structural_type': 'numpy.int64', 'name': 'ground_truth'},
		}])

		hyperparams_class = AutoCorrelation.AutoCorrelation.metadata.get_hyperparams().defaults()
		hyperparams_class = hyperparams_class.replace({'nlags': 2})
		#hyperparams_class = hyperparams_class.replace({'use_semantic_types': True})
		primitive = AutoCorrelation.AutoCorrelation(hyperparams=hyperparams_class)
		new_main = primitive.produce(inputs=main).value
		
		print(new_main)
		
		new_main_drop = new_main['value_acf']
		new_main_drop = new_main_drop.reset_index(drop = True)


		expected_result = pd.DataFrame({'acf':[1.000000, 0.700000, 0.412121, 0.148485, -0.078788, -0.257576, -0.375758, -0.421212, -0.381818, -0.245455]})
     	
		new_main_drop.reset_index()

		self.assertEqual(all(new_main_drop), all(expected_result))


		#print(main.metadata.to_internal_simple_structure())
		#print(new_main.metadata.to_internal_simple_structure())

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
					'length': 10,
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
			'metadata': {'structural_type': 'numpy.int64', 'name': 'd3mIndex'},
		}, {
			'selector': ['__ALL_ELEMENTS__', 1],
			'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
		}, {
			'selector': ['__ALL_ELEMENTS__', 2],
			'metadata': {'structural_type': 'numpy.float64', 'name': 'value'},
		}, {
			'selector': ['__ALL_ELEMENTS__', 3],
			'metadata': {'structural_type': 'numpy.int64', 'name': 'ground_truth'},
		}])

		params = primitive.get_params()
		primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
