import unittest

from d3m import container, utils as d3m_utils
from d3m.base import utils
from d3m.metadata import base as metadata_base


class TestBaseUtils(unittest.TestCase):
    def test_combine_columns_compact_metadata(self):
        main = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9], 'd1': [10, 11, 12], 'e1': [13, 14, 15]}, {
            'top_level': 'main',
        }, generate_metadata=False)
        main.metadata = main.metadata.generate(main, compact=True)
        main.metadata = main.metadata.update_column(0, {'name': 'aaa111'})
        main.metadata = main.metadata.update_column(1, {'name': 'bbb111', 'extra': 'b_column'})
        main.metadata = main.metadata.update_column(2, {'name': 'ccc111'})

        columns2 = container.DataFrame({'a2': [21, 22, 23], 'b2': [24, 25, 26]}, {
            'top_level': 'columns2',
        }, generate_metadata=False)
        columns2.metadata = columns2.metadata.generate(columns2, compact=True)
        columns2.metadata = columns2.metadata.update_column(0, {'name': 'aaa222'})
        columns2.metadata = columns2.metadata.update_column(1, {'name': 'bbb222'})

        columns3 = container.DataFrame({'a3': [31, 32, 33], 'b3': [34, 35, 36]}, {
            'top_level': 'columns3',
        }, generate_metadata=False)
        columns3.metadata = columns3.metadata.generate(columns3, compact=True)
        columns3.metadata = columns3.metadata.update_column(0, {'name': 'aaa333'})
        columns3.metadata = columns3.metadata.update_column(1, {'name': 'bbb333'})

        result = utils.combine_columns(main, [1, 2], [columns2, columns3], return_result='append', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [1, 4, 7, 10, 13, 21, 24, 31, 34],
            [2, 5, 8, 11, 14, 22, 25, 32, 35],
            [3, 6, 9, 12, 15, 23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 9,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'ccc111',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'd1',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'e1',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 7],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 8],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [1, 2], [columns2, columns3], return_result='new', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [21, 24, 31, 34],
            [22, 25, 32, 35],
            [23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'columns2',
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
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa222',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb222',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [1, 2], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [1, 21, 24, 31, 34, 10, 13],
            [2, 22, 25, 32, 35, 11, 14],
            [3, 23, 26, 33, 36, 12, 15],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 7,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'd1',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'name': 'e1',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [0, 1, 2, 3, 4], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [21, 24, 31, 34],
            [22, 25, 32, 35],
            [23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [4], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [1, 4, 7, 10, 21, 24, 31, 34],
            [2, 5, 8, 11, 22, 25, 32, 35],
            [3, 6, 9, 12, 23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 8,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'ccc111',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'd1',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'aaa222',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'bbb222',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'aaa333',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 7],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'bbb333',
            },
        }])

        result = utils.combine_columns(main, [0, 2, 4], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [21, 4, 24, 10, 31, 34],
            [22, 5, 25, 11, 32, 35],
            [23, 6, 26, 12, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'd1',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])
    def test_combine_columns_noncompact_metadata(self):
        main = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9], 'd1': [10, 11, 12], 'e1': [13, 14, 15]}, {
            'top_level': 'main',
        }, generate_metadata=False)
        main.metadata = main.metadata.generate(main, compact=False)
        main.metadata = main.metadata.update_column(0, {'name': 'aaa111'})
        main.metadata = main.metadata.update_column(1, {'name': 'bbb111', 'extra': 'b_column'})
        main.metadata = main.metadata.update_column(2, {'name': 'ccc111'})

        columns2 = container.DataFrame({'a2': [21, 22, 23], 'b2': [24, 25, 26]}, {
            'top_level': 'columns2',
        }, generate_metadata=False)
        columns2.metadata = columns2.metadata.generate(columns2, compact=False)
        columns2.metadata = columns2.metadata.update_column(0, {'name': 'aaa222'})
        columns2.metadata = columns2.metadata.update_column(1, {'name': 'bbb222'})

        columns3 = container.DataFrame({'a3': [31, 32, 33], 'b3': [34, 35, 36]}, {
            'top_level': 'columns3',
        }, generate_metadata=False)
        columns3.metadata = columns3.metadata.generate(columns3, compact=False)
        columns3.metadata = columns3.metadata.update_column(0, {'name': 'aaa333'})
        columns3.metadata = columns3.metadata.update_column(1, {'name': 'bbb333'})

        result = utils.combine_columns(main, [1, 2], [columns2, columns3], return_result='append', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [1, 4, 7, 10, 13, 21, 24, 31, 34],
            [2, 5, 8, 11, 14, 22, 25, 32, 35],
            [3, 6, 9, 12, 15, 23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 9,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'ccc111',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'd1',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'e1',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 7],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 8],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [1, 2], [columns2, columns3], return_result='new', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [21, 24, 31, 34],
            [22, 25, 32, 35],
            [23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'columns2',
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
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [1, 2], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [1, 21, 24, 31, 34, 10, 13],
            [2, 22, 25, 32, 35, 11, 14],
            [3, 23, 26, 33, 36, 12, 15],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 7,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'd1',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'name': 'e1',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [0, 1, 2, 3, 4], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [21, 24, 31, 34],
            [22, 25, 32, 35],
            [23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

        result = utils.combine_columns(main, [4], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [1, 4, 7, 10, 21, 24, 31, 34],
            [2, 5, 8, 11, 22, 25, 32, 35],
            [3, 6, 9, 12, 23, 26, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
                    'length': 8,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'aaa111',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'ccc111',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'd1',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'aaa222',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'bbb222',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'aaa333',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 7],
            'metadata': {
                'structural_type': 'numpy.int64',
                'name': 'bbb333',
            },
        }])

        result = utils.combine_columns(main, [0, 2, 4], [columns2, columns3], return_result='replace', add_index_columns=False)

        self.assertEqual(result.values.tolist(), [
            [21, 4, 24, 10, 31, 34],
            [22, 5, 25, 11, 32, 35],
            [23, 6, 26, 12, 33, 36],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
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
            'metadata': {
                'name': 'aaa222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'bbb222',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'd1',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'aaa333',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'bbb333',
                'structural_type': 'numpy.int64',
            },
        }])

    def test_combine_columns_new_with_index_compact_metadata(self):
        main = container.DataFrame({'d3mIndex': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9]}, columns=['d3mIndex', 'b1', 'c1'], generate_metadata=False)
        main.metadata = main.metadata.generate(main, compact=True)
        main.metadata = main.metadata.update_column(0, {'name': 'd3mIndex', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        main.metadata = main.metadata.update_column(1, {'name': 'b1', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']})
        main.metadata = main.metadata.update_column(2, {'name': 'c1', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']})

        columns = container.DataFrame({'d3mIndex': [1, 2, 3], 'b2': [4, 5, 6]}, columns=['d3mIndex', 'b2'], generate_metadata=False)
        columns.metadata = columns.metadata.generate(columns, compact=True)
        columns.metadata = columns.metadata.update_column(0, {'name': 'd3mIndex', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        columns.metadata = columns.metadata.update_column(1, {'name': 'b2', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']})

        result = utils.combine_columns(main, [], [columns], return_result='new', add_index_columns=True)

        self.assertEqual(result.values.tolist(), [
            [1, 4],
            [2, 5],
            [3, 6],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'b2',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
            },
        }])

    def test_combine_columns_new_with_index_noncompact_metadata(self):
        main = container.DataFrame({'d3mIndex': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9]}, columns=['d3mIndex', 'b1', 'c1'], generate_metadata=False)
        main.metadata = main.metadata.generate(main, compact=False)
        main.metadata = main.metadata.update_column(0, {'name': 'd3mIndex', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        main.metadata = main.metadata.update_column(1, {'name': 'b1', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']})
        main.metadata = main.metadata.update_column(2, {'name': 'c1', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']})

        columns = container.DataFrame({'d3mIndex': [1, 2, 3], 'b2': [4, 5, 6]}, columns=['d3mIndex', 'b2'], generate_metadata=False)
        columns.metadata = columns.metadata.generate(columns, compact=False)
        columns.metadata = columns.metadata.update_column(0, {'name': 'd3mIndex', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        columns.metadata = columns.metadata.update_column(1, {'name': 'b2', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute']})

        result = utils.combine_columns(main, [], [columns], return_result='new', add_index_columns=True)

        self.assertEqual(result.values.tolist(), [
            [1, 4],
            [2, 5],
            [3, 6],
        ])

        self.assertEqual(d3m_utils.to_json_structure(result.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
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
            'metadata': {
                'name': 'd3mIndex',
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'b2',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.int64',
            },
        }])


if __name__ == '__main__':
    unittest.main()
