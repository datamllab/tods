import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.data_processing import ContinuityValidation


class ContinuityValidationTest(unittest.TestCase):
    def test_basic(self):
        main = container.DataFrame({'d3mIndex': [0, 1, 2], 'timestamp': [1., 2., 4.], 'a': [1., 2., 3.], 'b': [2., 3., 4.], 'ground_truth': [0, 0, 0],},
                                    columns=['d3mIndex', 'timestamp', 'a', 'b', 'ground_truth'],
                                    generate_metadata=True)

        # print(main)

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
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'd3mIndex'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'}
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'}
            }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'ground_truth'}
        }])


        self.assertIsInstance(main, container.DataFrame)


        hyperparams_class = ContinuityValidation.ContinuityValidationPrimitive.metadata.get_hyperparams()
        primitive = ContinuityValidation.ContinuityValidationPrimitive(hyperparams=hyperparams_class.defaults())
        new_main = primitive.produce(inputs=main).value
    

        expected_output = container.DataFrame({'d3mIndex': [0, 1, 2, 3],
                                                'timestamp': [1., 2., 3., 4.],
                                                'a': [1., 2., 2.5, 3.], 'b': [2., 3., 3.5, 4.],
                                                'ground_truth': [0, 0, 0, 0]})
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
                    'length': 4,
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
                'name': 'd3mIndex',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'timestamp',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'a',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'b',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'ground_truth',
                'structural_type': 'numpy.int64',
            },
        }])

        self._test_continuity(new_main)

        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams.replace({'continuity_option': 'ablation'})
        primitive2 = ContinuityValidation.ContinuityValidationPrimitive(hyperparams=hyperparams)
        new_main2 = primitive2.produce(inputs=main).value
        print(new_main2)

        self.assertEqual(utils.to_json_structure(new_main2.metadata.to_internal_simple_structure()), [{
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
                'name': 'd3mIndex',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'timestamp',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'a',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'b',
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'ground_truth',
                'structural_type': 'numpy.int64',
            },
        }])



    def _test_continuity(self, data_value):
        tmp_col = data_value['timestamp']
        interval = tmp_col[1] - tmp_col[0]
        for i in range(2, tmp_col.shape[0]):
            self.assertEqual(interval, tmp_col[i] - tmp_col[i-1])



if __name__ == '__main__':
    unittest.main()
