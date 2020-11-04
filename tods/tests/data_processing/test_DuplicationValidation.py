import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.data_processing import DuplicationValidation


class DuplicationValidationTest(unittest.TestCase):
    def test_basic(self):
        main = container.DataFrame({'timestamp': [1., 1., 4.],'a': [1., 2., 3.], 'b': [2., 3., 4.],},
                                    columns=['timestamp', 'a', 'b'],
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
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'}
        }])


        self.assertIsInstance(main, container.DataFrame)


        hyperparams_class = DuplicationValidation.DuplicationValidationPrimitive.metadata.get_hyperparams()
        primitive = DuplicationValidation.DuplicationValidationPrimitive(hyperparams=hyperparams_class.defaults())
        new_main = primitive.produce(inputs=main).value
        print(new_main)

        expected_output = container.DataFrame({'timestamp': [1., 4.],'a': [1., 3.], 'b': [2., 4.],})
        self.assertEqual(new_main.values.tolist() , expected_output.values.tolist())        


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
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'timestamp',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'a',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'b',
                'structural_type': 'numpy.float64',
            },
        }])

        self._test_drop_duplication(new_main)

        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams.replace({'keep_option': 'average'})
        primitive2 = DuplicationValidation.DuplicationValidationPrimitive(hyperparams=hyperparams)
        new_main2 = primitive2.produce(inputs=main).value
        print(new_main2)

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
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'timestamp',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'a',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'b',
                'structural_type': 'numpy.float64',
            },
        }])


    def _test_drop_duplication(self, data_value):
        self.assertEqual(True in list(data_value.duplicated('timestamp')), False)



if __name__ == '__main__':
    unittest.main()
