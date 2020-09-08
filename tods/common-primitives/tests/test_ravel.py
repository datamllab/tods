import unittest

from d3m import container, utils

from common_primitives import ravel


class RavelAsRowPrimitiveTestCase(unittest.TestCase):
    def _get_data(self):
        data = container.DataFrame({
            'a': [1, 2, 3],
            'b': [container.ndarray([2, 3, 4]), container.ndarray([5, 6, 7]), container.ndarray([8, 9, 10])]
        }, {
            'top_level': 'foobar1',
        }, generate_metadata=True)

        data.metadata = data.metadata.update_column(1, {
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
        })

        return data

    def test_basic(self):
        dataframe = container.DataFrame({
            'a': [1, 2, 3],
            'b': ['a', 'b', 'c']
        }, {
            'top_level': 'foobar1',
        }, generate_metadata=True)

        self.assertEqual(dataframe.shape, (3, 2))

        for row_index in range(len(dataframe)):
            for column_index in range(len(dataframe.columns)):
                dataframe.metadata = dataframe.metadata.update((row_index, column_index), {
                    'location': (row_index, column_index),
                })

        dataframe.metadata.check(dataframe)

        hyperparams = ravel.RavelAsRowPrimitive.metadata.get_hyperparams()
        primitive = ravel.RavelAsRowPrimitive(hyperparams=hyperparams.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        self.assertEqual(dataframe.shape, (1, 6))

        self.assertEqual(dataframe.values.tolist(), [[1, 'a', 2, 'b', 3, 'c']])
        self.assertEqual(list(dataframe.columns), ['a', 'b', 'a', 'b', 'a', 'b'])

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 1,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'structural_type': 'd3m.container.pandas.DataFrame',
                'top_level': 'foobar1',
            },
        },
        {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 6,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        },
        {
            'selector': [0, 0],
            'metadata': {
                'location': [0, 0],
                'name': 'a',
                'structural_type': 'numpy.int64',
            },
        },
        {
            'selector': [0, 1],
            'metadata': {
                'location': [0, 1],
                'name': 'b',
                'structural_type': 'str',
            },
        },
        {
            'selector': [0, 2],
            'metadata': {
                'location': [1, 0],
                'name': 'a',
                'structural_type': 'numpy.int64',
            },
        },
        {
            'selector': [0, 3],
            'metadata': {
                'location': [1, 1],
                'name': 'b',
                'structural_type': 'str',
            },
        },
        {
            'selector': [0, 4],
            'metadata': {
                'location': [2, 0],
                'name': 'a',
                'structural_type': 'numpy.int64',
            },
        },
        {
            'selector': [0, 5],
            'metadata': {
                'location': [2, 1],
                'name': 'b',
                'structural_type': 'str',
            },
        }])


if __name__ == '__main__':
    unittest.main()
