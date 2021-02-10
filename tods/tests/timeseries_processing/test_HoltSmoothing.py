import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base


from tods.timeseries_processing import HoltSmoothing
import pandas as pd


class HoltSmoothingTestCase(unittest.TestCase):
    def test_basic(self):
        main = container.DataFrame({'timestamp': [1, 2, 3], 'value': [1,2,3]}, {
            'top_level': 'main',
        },  generate_metadata=True)
       

        self.assertEqual(utils.to_json_structure(main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'main',
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
            'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'value'},
        }])

        hyperparams_class = HoltSmoothing.HoltSmoothingPrimitive.metadata.get_hyperparams()
        primitive = HoltSmoothing.HoltSmoothingPrimitive(hyperparams=hyperparams_class.defaults())
        primitive.set_training_data(inputs=main)
        primitive.fit()
        output_main = primitive.produce(inputs=main).value
        output_main = round(output_main, 2) 

        expected_result = container.DataFrame(data = { 'timestamp' : [1,2,3], 'value_holt_smoothing': [2.00,2.76,3.54]})
   
        
        self.assertEqual(output_main[['timestamp','value_holt_smoothing']].values.tolist(), expected_result[['timestamp','value_holt_smoothing']].values.tolist())

        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
