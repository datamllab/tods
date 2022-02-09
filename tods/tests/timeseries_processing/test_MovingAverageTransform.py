import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base


from tods.timeseries_processing import MovingAverageTransformer
import pandas as pd


class MovingAverageTransformTestCase(unittest.TestCase):
    def test_basic(self):
        main = container.DataFrame({'timestamp': [20201, 20202, 20203,20204,20205], 'value': [100,200,300,400,500],}, {
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
                    'length': 5,
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

        hyperparams_class = MovingAverageTransformer.MovingAverageTransformerPrimitive.metadata.get_hyperparams()
        primitive = MovingAverageTransformer.MovingAverageTransformerPrimitive(hyperparams=hyperparams_class.defaults())
        primitive.set_training_data(inputs=main)
        primitive.fit()
        output_main = primitive.produce(inputs=main).value
       

        expected_result = container.DataFrame(data = { 'timestamp' : [20201,20202,20203,20204,20205], 'value': [150.0,200.0,300.0,400.0,450.0]})
        print ("expected_result", expected_result)


        self.assertEqual(output_main[['timestamp','value_moving_average']].values.tolist(), expected_result[['timestamp','value']].values.tolist())
        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
