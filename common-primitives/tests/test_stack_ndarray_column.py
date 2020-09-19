import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import stack_ndarray_column


class StackNDArrayColumnPrimitiveTestCase(unittest.TestCase):
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
        data = self._get_data()

        data_metadata_before = data.metadata.to_internal_json_structure()

        stack_hyperparams_class = stack_ndarray_column.StackNDArrayColumnPrimitive.metadata.get_hyperparams()
        stack_primitive = stack_ndarray_column.StackNDArrayColumnPrimitive(hyperparams=stack_hyperparams_class.defaults())
        stack_array = stack_primitive.produce(inputs=data).value

        self.assertEqual(stack_array.shape, (3, 3))

        self._test_metadata(stack_array.metadata)

        self.assertEqual(data.metadata.to_internal_json_structure(), data_metadata_before)

    def _test_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'foobar1',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
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
                    'length': 3,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
                # It is unclear if name and semantic types should be moved to rows, but this is what currently happens.
                'name': 'b',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])


if __name__ == '__main__':
    unittest.main()
