

import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.data_processing import TimeStampValidation


class TimeStampValidationTestCase(unittest.TestCase):
    def test_basic(self):

        main = container.DataFrame({'timestamp': [1,3,2,5], 'a': [1.0,2.0,3.0,4.0],'b':[1.0,4.0,5.0,6.0]},columns=['timestamp', 'a', 'b'],
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
                    'length': 4,
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
            'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        }])

        hyperparams_class = TimeStampValidation.TimeStampValidationPrimitive.metadata.get_hyperparams()
        primitive = TimeStampValidation.TimeStampValidationPrimitive(hyperparams=hyperparams_class.defaults())
        output_main = primitive.produce(inputs=main).value
        print(output_main[['timestamp','a','b']].values.tolist())
        expected_output = container.DataFrame({'timestamp': [1,2,3,5], 'a': [1.0,3.0,2.0,4.0],'b': [1.0,5.0,4.0,6.0]})

        self.assertEqual(output_main[['timestamp','a','b']].values.tolist() , expected_output[['timestamp','a','b']].values.tolist())

        self.assertEqual(utils.to_json_structure(output_main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 4,
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
            'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        }])

        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()

