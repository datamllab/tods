import unittest


from datetime import datetime

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.data_processing import TimeIntervalTransform


#import utils as test_utils

import numpy as np

class TimeIntervalTransformTestCase(unittest.TestCase):
	def test_basic(self):
		self.maxDiff = None
		main = container.DataFrame({'d3mIndex': [0, 1, 2, 3, 4, 5, 6, 7], 
									'timestamp': [1472918400, 1472918700, 1472919000, 1472919300,
													1472919600, 1472919900, 1472920200, 1472920500],
									'value': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 
									'ground_truth': [0, 1, 0, 1, 0, 1, 0,1]},
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
					'length': 8,
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

		hyperparams_class = TimeIntervalTransform.TimeIntervalTransformPrimitive.metadata.get_hyperparams()
		primitive = TimeIntervalTransform.TimeIntervalTransformPrimitive(hyperparams=hyperparams_class.defaults())
		new_main = primitive.produce(inputs=main).value
		new_rows = len(new_main.index)
		self.assertEqual(new_rows, 8)

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
					'length': 8,
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

