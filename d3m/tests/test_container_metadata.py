import collections
import unittest

import numpy
import pandas

from d3m import container, utils
from d3m.metadata import base


class TestContainerMetadata(unittest.TestCase):
    def test_update_with_generated_metadata(self):
        metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        })

        cells_metadata = collections.OrderedDict()
        cells_metadata[('a',)] = {'other': 1}
        cells_metadata[('b',)] = {'other': 2}
        cells_metadata[('c',)] = {'other': 3}
        cells_metadata[(base.ALL_ELEMENTS,)] = {'foo': 'bar'}
        cells_metadata[('other', 'a')] = {'other': 4}
        cells_metadata[('other', 'b')] = {'other': 5}
        cells_metadata[('other', 'c')] = {'other': 6}
        cells_metadata[('other', base.ALL_ELEMENTS)] = {'foo': 'bar2'}

        metadata._update_with_generated_metadata(cells_metadata)

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'foo': 'bar'},
        }, {
            'selector': ['a'],
            'metadata': {'other': 1},
        }, {
            'selector': ['b'],
            'metadata': {'other': 2},
        }, {
            'selector': ['c'],
            'metadata': {'other': 3},
        }, {
            'selector': ['other', '__ALL_ELEMENTS__'],
            'metadata': {'foo': 'bar2'},
        }, {
            'selector': ['other', 'a'],
            'metadata': {'other': 4},
        }, {
            'selector': ['other', 'b'],
            'metadata': {'other': 5},
        }, {
            'selector': ['other', 'c'],
            'metadata': {'other': 6},
        }])

        metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
            'semantic_types': ['http://example.com/Type1'],
            'dimension': {
                'length': 0,
                'foobar': 42,
                'semantic_types': ['http://example.com/Type2'],
            }
        })

        metadata = metadata.update(('a',), {
            'semantic_types': ['http://example.com/Type3'],
            'dimension': {
                'length': 0,
                'foobar': 45,
                'semantic_types': ['http://example.com/Type4'],
            }
        })

        cells_metadata = collections.OrderedDict()
        cells_metadata[()] = {
            'other': 1,
            'structural_type': container.ndarray,
            'semantic_types': ['http://example.com/Type1a'],
            'dimension': {
                'length': 100,
                'name': 'test1',
                'semantic_types': ['http://example.com/Type2a'],
            }
        }
        cells_metadata[('a',)] = {
            'semantic_types': ['http://example.com/Type3', 'http://example.com/Type3a'],
            'dimension': {
                'length': 200,
                'name': 'test2',
                'semantic_types': ['http://example.com/Type4', 'http://example.com/Type4a'],
            }
        }
        cells_metadata[('b',)] = {'other': 2}

        metadata._update_with_generated_metadata(cells_metadata)

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'other': 1,
                'semantic_types': ['http://example.com/Type1', 'http://example.com/Type1a'],
                'dimension': {
                    'length': 100,
                    'name': 'test1',
                    'foobar': 42,
                    'semantic_types': ['http://example.com/Type2', 'http://example.com/Type2a'],
                },
            },
        }, {
            'selector': ['a'],
            'metadata': {
                'semantic_types': ['http://example.com/Type3', 'http://example.com/Type3a'],
                'dimension': {
                    'length': 200,
                    'name': 'test2',
                    'foobar': 45,
                    'semantic_types': ['http://example.com/Type4', 'http://example.com/Type4a'],
                },
            },
        }, {
            'selector': ['b'],
            'metadata': {'other': 2},
        }])

        self.assertEqual(metadata.to_json_structure(), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'other': 1,
                'semantic_types': ['http://example.com/Type1', 'http://example.com/Type1a'],
                'dimension': {
                    'length': 100,
                    'name': 'test1',
                    'foobar': 42,
                    'semantic_types': ['http://example.com/Type2', 'http://example.com/Type2a'],
                },
            },
        }, {
            'selector': ['a'],
            'metadata': {
                'semantic_types': ['http://example.com/Type3', 'http://example.com/Type3a'],
                'dimension': {
                    'length': 200,
                    'name': 'test2',
                    'foobar': 45,
                    'semantic_types': ['http://example.com/Type4', 'http://example.com/Type4a'],
                },
            },
        }, {
            'selector': ['b'],
            'metadata': {'other': 2},
        }])

    def test_dataframe(self):
        df = container.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']}, generate_metadata=True)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                'name': 'A',
                'structural_type': 'numpy.int64',
            }
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'str',
            },
        }])

    def test_dataset(self):
        dataset = container.Dataset({'0': container.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})}, generate_metadata=False)

        compact_metadata = dataset.metadata.generate(dataset, compact=True)
        noncompact_metadata = dataset.metadata.generate(dataset, compact=False)

        self.assertEqual(utils.to_json_structure(compact_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'dimension': {
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'str',
            },
        }])

        self.assertEqual(utils.to_json_structure(noncompact_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'dimension': {
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['0'],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['0', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 2,
                },
            },
        }, {
            'selector': ['0', '__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['0', '__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'str',
            },
        }])

    def test_list(self):
        lst = container.List(['a', 'b', 'c'], generate_metadata=True)

        self.assertEqual(utils.to_json_structure(lst.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'str',
            },
        }])

        lst = container.List([1, 'a', 2.0], generate_metadata=True)

        self.assertEqual(utils.to_json_structure(lst.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': [0],
            'metadata': {
                'structural_type': 'int',
            },
        }, {
            'selector': [1],
            'metadata': {
                'structural_type': 'str',
            },
        }, {
            'selector': [2],
            'metadata': {
                'structural_type': 'float',
            },
        }])

        lst = container.List([container.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})], generate_metadata=True)

        self.assertEqual(utils.to_json_structure(lst.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'str',
            },
        }])

    def test_ndarray(self):
        array = container.ndarray(numpy.array([1, 2, 3]), generate_metadata=True)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

    def test_dataframe_with_names_kept(self):
        df = container.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']}, generate_metadata=True)

        df.metadata = df.metadata.update((base.ALL_ELEMENTS, 0), {
            'name': 'first_column',
        })
        df.metadata = df.metadata.update((base.ALL_ELEMENTS, 1), {
            'name': 'second_column',
        })

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                'name': 'first_column',
                'structural_type': 'numpy.int64',
            }
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'second_column',
                'structural_type': 'str',
            },
        }])

        df2 = container.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd']})

        df2.metadata = df.metadata.generate(df2)

        self.assertEqual(utils.to_json_structure(df2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'first_column',
                'structural_type': 'numpy.int64',
            }
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'second_column',
                'structural_type': 'str',
            },
        }])

    def test_dataframe_tabular_semantic_types(self):
        # A DataFrame with explicit WRONG metadata.
        df = container.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']}, {
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
            },
        }, generate_metadata=True)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    # We respect the name, but we override the semantic types.
                    'name': 'columns',
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
                'name': 'A',
                'structural_type': 'numpy.int64',
            }
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'str',
            },
        }])

    def test_complex_value(self):
        dataset = container.Dataset({
            '0': container.DataFrame({
                'A': [
                    container.ndarray(numpy.array(['a', 'b', 'c'])),
                    container.ndarray(numpy.array([1, 2, 3])),
                    container.ndarray(numpy.array([1.0, 2.0, 3.0])),
                ],
                'B': [
                    container.List(['a', 'b', 'c']),
                    container.List([1, 2, 3]),
                    container.List([1.0, 2.0, 3.0]),
                ],
            }),
        }, generate_metadata=False)

        dataset_metadata = dataset.metadata.generate(dataset, compact=True)

        self.assertEqual(utils.to_json_structure(dataset_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'dimension': {
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 3
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 0],
            'metadata': {
                'structural_type': 'd3m.container.numpy.ndarray',
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 1],
            'metadata': {
                'structural_type': 'd3m.container.list.List',
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0, 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.str_',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0, 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'str',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1, 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1, 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2, 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2, 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'float',
            }
        }])

        dataset_metadata = dataset.metadata.generate(dataset, compact=False)

        self.assertEqual(utils.to_json_structure(dataset_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
               'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
               'structural_type': 'd3m.container.dataset.Dataset',
            },
        }, {
            'selector': ['0'],
            'metadata': {
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
               'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
               'structural_type': 'd3m.container.pandas.DataFrame',
            },
        },
        {
            'selector': ['0', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 2,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        },
        {
            'selector': ['0', '__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        },
        {
            'selector': ['0', '__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        },
        {
            'selector': ['0', 0, 0],
            'metadata': {
                'dimension': {
                    'length': 3,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        },
        {
            'selector': ['0', 0, 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.str_'
            },
        },
        {
            'selector': ['0', 0, 1],
            'metadata': {
                'dimension': {
                    'length': 3,
                },
                'structural_type': 'd3m.container.list.List',
            },
        }, {
            'selector': ['0', 0, 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'str',
            },
        }, {
            'selector': ['0', 1, 0],
            'metadata': {
                'dimension': {
                    'length': 3,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['0', 1, 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['0', 1, 1],
            'metadata': {
                'dimension': {
                    'length': 3,
                },
                'structural_type': 'd3m.container.list.List',
            },
        }, {
            'selector': ['0', 1, 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }, {
            'selector': ['0', 2, 0],
            'metadata': {
                'dimension': {
                    'length': 3,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['0', 2, 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.float64',
            },
        },
        {
            'selector': ['0', 2, 1],
            'metadata': {
                'dimension': {
                    'length': 3,
                },
                'structural_type': 'd3m.container.list.List',
            },
        }, {
            'selector': ['0', 2, 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'float',
            },
        }])

    def test_dataframe_with_objects(self):
        df = pandas.DataFrame({str(i): [str(j) for j in range(10)] for i in range(5)}, columns=[str(i) for i in range(5)])

        df = container.DataFrame(df, generate_metadata=False)

        compact_metadata = df.metadata.generate(df, compact=True)
        noncompact_metadata = df.metadata.generate(df, compact=False)

        basic_metadata = [
            {
                'selector': [],
                'metadata': {
                    'schema': base.CONTAINER_SCHEMA_VERSION,
                    'structural_type': 'd3m.container.pandas.DataFrame',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                    'dimension': {
                        'name': 'rows',
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        'length': 10,
                    },
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__'],
                'metadata': {
                    'dimension': {
                        'name': 'columns',
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        'length': 5,
                    },
                },
            },
        ]

        column_names = [{'selector': ['__ALL_ELEMENTS__', i], 'metadata': {'name': str(i)}} for i in range(5)]

        self.assertEqual(utils.to_json_structure(compact_metadata.to_internal_simple_structure()), basic_metadata + [
            {
                'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'str',
                },
            }
        ] + column_names)

        column_names = [{'selector': ['__ALL_ELEMENTS__', i], 'metadata': {'name': str(i), 'structural_type': 'str'}} for i in range(5)]

        self.assertEqual(utils.to_json_structure(noncompact_metadata.to_internal_simple_structure()), basic_metadata + column_names)

    def test_list_with_objects(self):
        l = container.List([container.List([str(j) for i in range(5)]) for j in range(10)], generate_metadata=True)

        self.assertEqual(utils.to_json_structure(l.metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'schema': base.CONTAINER_SCHEMA_VERSION,
                    'structural_type': 'd3m.container.list.List',
                    'dimension': {
                        'length': 10,
                    },
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'd3m.container.list.List',
                    'dimension': {
                        'length': 5,
                    },
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'str',
                },
            },
        ])

    def test_ndarray_with_objects(self):
        array = numpy.array([[[str(k) for k in range(5)] for i in range(10)] for j in range(10)], dtype=object)

        array = container.ndarray(array, generate_metadata=True)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'schema': base.CONTAINER_SCHEMA_VERSION,
                    'structural_type': 'd3m.container.numpy.ndarray',
                    'dimension': {
                        'length': 10,
                    },
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__'],
                'metadata': {
                    'dimension': {
                        'length': 10,
                    },
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                'metadata': {
                    'dimension': {
                        'length': 5,
                    },
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'str',
                },
            },
        ])

    def test_dict_with_objects(self):
        l = container.List([{str(i): {str(j): j for j in range(10)} for i in range(5)}], generate_metadata=True)

        self.assertEqual(utils.to_json_structure(l.metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'schema': base.CONTAINER_SCHEMA_VERSION,
                    'structural_type': 'd3m.container.list.List',
                    'dimension': {
                        'length': 1,
                    },
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'dict',
                    'dimension': {
                        'length': 5,
                    },
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'dict',
                    'dimension': {
                        'length': 10.
                    },
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
                'metadata': {
                    'structural_type': 'int',
                },
            },
        ])

    def test_custom_column_name_with_compacting(self):
        dataframe = container.DataFrame({'a': ['1.0', '2.0', '3.0']}, generate_metadata=False)

        dataframe.metadata = dataframe.metadata.generate(dataframe, compact=True)

        dataframe.metadata = dataframe.metadata.update((base.ALL_ELEMENTS, 0), {
            'name': 'test',
            'foo': 'bar',
        })

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 1,
                }
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'str',
                'name': 'a',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'test',
                'foo': 'bar',
            },
        }])

        dataframe.metadata = dataframe.metadata.generate(dataframe, compact=True)

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 1,
                }
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'str',
                'name': 'a',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'test',
                'foo': 'bar',
            },
        }])

        dataframe.metadata = dataframe.metadata.update((base.ALL_ELEMENTS, 0), {
            'name': base.NO_VALUE,
        })

        dataframe.metadata = dataframe.metadata.generate(dataframe, compact=True)

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 1,
                }
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'str',
                'name': 'a',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': '__NO_VALUE__',
                'foo': 'bar',
            },
        }])


    def test_custom_column_name_without_compacting(self):
        dataframe = container.DataFrame({'a': ['1.0', '2.0', '3.0']}, generate_metadata=False)

        dataframe.metadata = dataframe.metadata.generate(dataframe, compact=False)

        dataframe.metadata = dataframe.metadata.update((base.ALL_ELEMENTS, 0), {
            'name': 'test',
            'foo': 'bar',
        })

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 1,
                }
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'test',
                'foo': 'bar',
                'structural_type': 'str',
            },
        }])

        dataframe.metadata = dataframe.metadata.generate(dataframe, compact=False)

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 1,
                }
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'test',
                'foo': 'bar',
                'structural_type': 'str',
            },
        }])

        dataframe.metadata = dataframe.metadata.update((base.ALL_ELEMENTS, 0), {
            'name': base.NO_VALUE,
        })

        dataframe.metadata = dataframe.metadata.generate(dataframe, compact=False)

        self.assertEqual(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                    'length': 1,
                }
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': '__NO_VALUE__',
                'foo': 'bar',
                'structural_type': 'str',
            },
        }])

    def test_unset_structural_type(self):
        dataframe = container.DataFrame({'a': ['a', 'b', 'c'], 'b': ['a', 'b', 'c']}, generate_metadata=False)

        compact_metadata = dataframe.metadata.generate(dataframe, compact=True)

        all_elements_metadata = compact_metadata.query((base.ALL_ELEMENTS, base.ALL_ELEMENTS))
        compact_metadata = compact_metadata.remove((base.ALL_ELEMENTS, base.ALL_ELEMENTS), strict_all_elements=True)
        compact_metadata = compact_metadata.update((base.ALL_ELEMENTS, 0), all_elements_metadata)
        compact_metadata = compact_metadata.update((base.ALL_ELEMENTS, 1), all_elements_metadata)

        compact_metadata = compact_metadata.generate(dataframe, compact=True)

        self.assertEqual(utils.to_json_structure(compact_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                'structural_type': 'str',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'a',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'b',
            },
        }])

        compact_metadata = dataframe.metadata.generate(dataframe, compact=False)

        all_elements_metadata = compact_metadata.query((base.ALL_ELEMENTS, base.ALL_ELEMENTS))
        compact_metadata = compact_metadata.remove((base.ALL_ELEMENTS, base.ALL_ELEMENTS), strict_all_elements=True)
        compact_metadata = compact_metadata.update((base.ALL_ELEMENTS, 0), all_elements_metadata)
        compact_metadata = compact_metadata.update((base.ALL_ELEMENTS, 1), all_elements_metadata)

        compact_metadata = compact_metadata.generate(dataframe, compact=False)

        self.assertEqual(utils.to_json_structure(compact_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': base.CONTAINER_SCHEMA_VERSION,
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
                'name': 'a',
                'structural_type': 'str',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'b',
                'structural_type': 'str',
            },
        }])


if __name__ == '__main__':
    unittest.main()
