import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.timeseries_processing import SKAxiswiseScaler
import numpy as np

class SKStandardizationTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff=None
        main = container.DataFrame({'a1': [1., 2., 3.], 'b1': [2., 3., 4.],
                                    'a2': [3., 4., 5.], 'c1': [4., 5., 6.],
                                    'a3': [5., 6., 7.], 'a1a': [6., 7., 8.]},
                                   # {'top_level': 'main', },
            columns=['a1', 'b1', 'a2', 'c1', 'a3', 'a1a'],
                                   generate_metadata=True)
        main.metadata = main.metadata.update_column(0, {'name': 'aaa111'})
        main.metadata = main.metadata.update_column(1, {'name': 'bbb111'})
        main.metadata = main.metadata.update_column(2, {'name': 'aaa222'})
        main.metadata = main.metadata.update_column(3, {'name': 'ccc111'})
        main.metadata = main.metadata.update_column(4, {'name': 'aaa333'})
        main.metadata = main.metadata.update_column(5, {'name': 'aaa111'})

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
                    'length': 6,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'aaa111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'bbb111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'aaa222'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'ccc111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'aaa333'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'aaa111'},
        }])

        hyperparams_class = SKAxiswiseScaler.SKAxiswiseScalerPrimitive.metadata.get_hyperparams()
        primitive = SKAxiswiseScaler.SKAxiswiseScalerPrimitive(hyperparams=hyperparams_class.defaults())
        new_main = primitive.produce(inputs=main).value
        new_mean, new_std = new_main.values.mean(0), new_main.values.std(0)

        mean_mse = np.matmul(new_mean.T, new_mean)
        std_mse = np.matmul((new_std - np.ones_like(new_std)).T, (new_std - np.ones_like(new_std)))

        # print(new_main)
        # print(mean_mse, std_mse)

        self.assertAlmostEqual(mean_mse.__float__(), 0., delta=1e-8)
        self.assertAlmostEqual(std_mse.__float__(), 0., delta=1e-8)

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
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'aaa222',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'ccc111',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'aaa333',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'aaa111',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }])

        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
