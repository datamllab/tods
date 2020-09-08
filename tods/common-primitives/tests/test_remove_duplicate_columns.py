import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import remove_duplicate_columns


class RemoveDuplicateColumnsPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        main = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'a2': [1, 2, 3], 'c1': [7, 8, 9], 'a3': [1, 2, 3], 'a1a': [1, 2, 3]}, {
            'top_level': 'main',
        }, columns=['a1', 'b1', 'a2', 'c1', 'a3', 'a1a'], generate_metadata=True)
        main.metadata = main.metadata.update_column(0, {'name': 'aaa111'})
        main.metadata = main.metadata.update_column(1, {'name': 'bbb111'})
        main.metadata = main.metadata.update_column(2, {'name': 'aaa222'})
        main.metadata = main.metadata.update_column(3, {'name': 'ccc111'})
        main.metadata = main.metadata.update_column(4, {'name': 'aaa333'})
        main.metadata = main.metadata.update_column(5, {'name': 'aaa111'})

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
                    'length': 6,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'aaa111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'bbb111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'aaa222'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'ccc111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'aaa333'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'aaa111'},
        }])

        hyperparams_class = remove_duplicate_columns.RemoveDuplicateColumnsPrimitive.metadata.get_hyperparams()
        primitive = remove_duplicate_columns.RemoveDuplicateColumnsPrimitive(hyperparams=hyperparams_class.defaults())
        primitive.set_training_data(inputs=main)
        primitive.fit()
        new_main = primitive.produce(inputs=main).value

        self.assertEqual(new_main.values.tolist(), [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        ])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
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
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
                'other_names': ['aaa222', 'aaa333'],
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'ccc111',
                'structural_type': 'numpy.int64',
            },
        }])

        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
