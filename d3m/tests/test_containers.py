import copy
import os.path
import pickle
import tempfile
import unittest
import warnings

import numpy
import pandas
import pandas.core.common

from d3m import container, utils
from d3m.container import utils as container_utils
from d3m.metadata import base as metadata_base

copy_functions = {
    'obj.copy()': lambda obj: obj.copy(),
    'obj[:]': lambda obj: obj[:],
    'copy.copy()': lambda obj: copy.copy(obj),
    'copy.deepcopy()': lambda obj: copy.deepcopy(obj),
    'pickle.loads(pickle.dumps())': lambda obj: pickle.loads(pickle.dumps(obj)),
}


class TestContainers(unittest.TestCase):
    def test_list(self):
        l = container.List()

        self.assertTrue(hasattr(l, 'metadata'))

        l = container.List([1, 2, 3], generate_metadata=True)

        l.metadata = l.metadata.update((), {
            'test': 'foobar',
        })

        self.assertSequenceEqual(l, [1, 2, 3])
        self.assertIsInstance(l, container.List)
        self.assertTrue(hasattr(l, 'metadata'))
        self.assertEqual(l.metadata.query(()).get('test'), 'foobar')

        self.assertIsInstance(l, container.List)
        self.assertIsInstance(l, list)

        self.assertNotIsInstance([], container.List)

        for name, copy_function in copy_functions.items():
            l_copy = copy_function(l)

            self.assertIsInstance(l_copy, container.List, name)
            self.assertTrue(hasattr(l_copy, 'metadata'), name)

            self.assertSequenceEqual(l, l_copy, name)
            self.assertEqual(l.metadata.to_internal_json_structure(), l_copy.metadata.to_internal_json_structure(), name)
            self.assertEqual(l_copy.metadata.query(()).get('test'), 'foobar', name)

        l_copy = container.List(l, {
            'test2': 'barfoo',
        }, generate_metadata=True)

        self.assertIsInstance(l_copy, container.List)
        self.assertTrue(hasattr(l_copy, 'metadata'))

        self.assertSequenceEqual(l, l_copy)
        self.assertEqual(l_copy.metadata.query(()), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
            'dimension': {
                'length': 3,
            },
            'test': 'foobar',
            'test2': 'barfoo',
        })

        self.assertEqual(l[1], 2)

        with self.assertRaisesRegex(TypeError, 'list indices must be integers or slices, not tuple'):
            l[1, 2]

        l_slice = l[1:3]

        self.assertSequenceEqual(l, [1, 2, 3])
        self.assertSequenceEqual(l_slice, [2, 3])
        self.assertIsInstance(l_slice, container.List)
        self.assertTrue(hasattr(l_slice, 'metadata'))
        self.assertEqual(l.metadata.to_internal_json_structure(), l_slice.metadata.to_internal_json_structure())

        l_added = l + [4, 5]

        self.assertSequenceEqual(l, [1, 2, 3])
        self.assertSequenceEqual(l_added, [1, 2, 3, 4, 5])
        self.assertIsInstance(l_added, container.List)
        self.assertTrue(hasattr(l_added, 'metadata'))
        self.assertEqual(l.metadata.to_internal_json_structure(), l_added.metadata.to_internal_json_structure())

        l_added += [6, 7]

        self.assertSequenceEqual(l_added, [1, 2, 3, 4, 5, 6, 7])
        self.assertIsInstance(l_added, container.List)
        self.assertTrue(hasattr(l_added, 'metadata'))
        self.assertEqual(l.metadata.to_internal_json_structure(), l_added.metadata.to_internal_json_structure())

        l_multiplied = l * 3

        self.assertSequenceEqual(l, [1, 2, 3])
        self.assertSequenceEqual(l_multiplied, [1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertIsInstance(l_multiplied, container.List)
        self.assertTrue(hasattr(l_multiplied, 'metadata'))
        self.assertEqual(l.metadata.to_internal_json_structure(), l_multiplied.metadata.to_internal_json_structure())

        l_multiplied = 3 * l

        self.assertSequenceEqual(l, [1, 2, 3])
        self.assertSequenceEqual(l_multiplied, [1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertIsInstance(l_multiplied, container.List)
        self.assertTrue(hasattr(l_multiplied, 'metadata'))
        self.assertEqual(l.metadata.to_internal_json_structure(), l_multiplied.metadata.to_internal_json_structure())

        l_multiplied *= 2

        self.assertSequenceEqual(l_multiplied, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertIsInstance(l_multiplied, container.List)
        self.assertTrue(hasattr(l_multiplied, 'metadata'))
        self.assertEqual(l.metadata.to_internal_json_structure(), l_multiplied.metadata.to_internal_json_structure())

    def test_ndarray(self):
        array = container.ndarray(numpy.array([1, 2, 3]), generate_metadata=True)
        self.assertTrue(numpy.array_equal(array, [1, 2, 3]))
        self.assertIsInstance(array, container.ndarray)
        self.assertTrue(hasattr(array, 'metadata'))

        self.assertIsInstance(array, numpy.ndarray)

        self.assertNotIsInstance(numpy.array([]), container.ndarray)

        array.metadata = array.metadata.update((), {
            'test': 'foobar',
        })

        self.assertEqual(array.metadata.query(()).get('test'), 'foobar')

        for name, copy_function in copy_functions.items():
            array_copy = copy_function(array)

            self.assertIsInstance(array_copy, container.ndarray, name)
            self.assertTrue(hasattr(array_copy, 'metadata'), name)

            self.assertTrue(numpy.array_equal(array, array_copy), name)
            self.assertEqual(array.metadata.to_internal_json_structure(), array_copy.metadata.to_internal_json_structure(), name)
            self.assertEqual(array_copy.metadata.query(()).get('test'), 'foobar', name)


        array_copy = container.ndarray(array, {
            'test2': 'barfoo',
        }, generate_metadata=True)

        self.assertIsInstance(array_copy, container.ndarray)
        self.assertTrue(hasattr(array_copy, 'metadata'))

        self.assertTrue(numpy.array_equal(array, array_copy))
        self.assertEqual(array_copy.metadata.query(()), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
            'dimension': {
                'length': 3,
            },
            'test': 'foobar',
            'test2': 'barfoo',
        })

        array_from_list = container.ndarray([1, 2, 3], generate_metadata=True)
        self.assertTrue(numpy.array_equal(array_from_list, [1, 2, 3]))
        self.assertIsInstance(array_from_list, container.ndarray)
        self.assertTrue(hasattr(array_from_list, 'metadata'))

    def test_dataframe_to_csv(self):
        df = container.DataFrame(pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), generate_metadata=True)
        df.metadata = df.metadata.update((metadata_base.ALL_ELEMENTS, 0), {'name': 'E'})
        df.metadata = df.metadata.update((metadata_base.ALL_ELEMENTS, 1), {'name': 'F'})

        self.assertEqual(df.columns.tolist(), ['A', 'B'])
        self.assertEqual(df.to_csv(), 'E,F\n1,4\n2,5\n3,6\n')

    def test_dataframe(self):
        df = container.DataFrame(pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}), generate_metadata=True)
        self.assertTrue(df._data.equals(pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})._data))
        self.assertIsInstance(df, container.DataFrame)
        self.assertTrue(hasattr(df, 'metadata'))

        self.assertIsInstance(df, pandas.DataFrame)

        self.assertNotIsInstance(pandas.DataFrame({'A': [1, 2, 3]}), container.DataFrame)

        df.metadata = df.metadata.update((), {
            'test': 'foobar',
        })

        self.assertEqual(df.metadata.query(()).get('test'), 'foobar')

        for name, copy_function in copy_functions.items():
            df_copy = copy_function(df)

            self.assertIsInstance(df_copy, container.DataFrame, name)
            self.assertTrue(hasattr(df_copy, 'metadata'), name)

            self.assertTrue(df.equals(df_copy), name)
            self.assertEqual(df.metadata.to_internal_json_structure(), df_copy.metadata.to_internal_json_structure(), name)
            self.assertEqual(df_copy.metadata.query(()).get('test'), 'foobar', name)

        df_copy = container.DataFrame(df, {
            'test2': 'barfoo',
        }, generate_metadata=True)

        self.assertIsInstance(df_copy, container.DataFrame)
        self.assertTrue(hasattr(df_copy, 'metadata'))

        self.assertTrue(numpy.array_equal(df, df_copy))
        self.assertEqual(df_copy.metadata.query(()), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
            'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Table',),
            'dimension': {
                'name': 'rows',
                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TabularRow',),
                'length': 3
            },
            'test': 'foobar',
            'test2': 'barfoo',
        })

        df_from_dict = container.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, generate_metadata=True)
        self.assertTrue(df_from_dict._data.equals(pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})._data))
        self.assertIsInstance(df_from_dict, container.DataFrame)
        self.assertTrue(hasattr(df_from_dict, 'metadata'))

        # Regression tests to make sure column name cannot overwrite DataFrame
        # attributes we use (like metadata and custom methods).
        dataframe = container.DataFrame({'metadata': [0], 'select_columns': [1]})
        self.assertIsInstance(dataframe.metadata, metadata_base.DataMetadata)
        self.assertIsInstance(dataframe.select_columns([0]), container.DataFrame)
        self.assertEqual(dataframe.loc[0, 'metadata'], 0)
        self.assertEqual(dataframe.loc[0, 'select_columns'], 1)

    def test_dataset(self):
        dataset = container.Dataset.load('sklearn://boston')

        self.assertIsInstance(dataset, container.Dataset)
        self.assertTrue(hasattr(dataset, 'metadata'))

        dataset.metadata = dataset.metadata.update((), {
            'test': 'foobar',
        })

        self.assertEqual(dataset.metadata.query(()).get('test'), 'foobar')

        for name, copy_function in copy_functions.items():
            # Not supported on dicts.
            if name == 'obj[:]':
                continue

            dataset_copy = copy_function(dataset)

            self.assertIsInstance(dataset_copy, container.Dataset, name)
            self.assertTrue(hasattr(dataset_copy, 'metadata'), name)

            self.assertEqual(len(dataset), len(dataset_copy), name)
            self.assertEqual(dataset.keys(), dataset_copy.keys(), name)
            for resource_name in dataset.keys():
                self.assertTrue(numpy.array_equal(dataset[resource_name], dataset_copy[resource_name]), name)
            self.assertEqual(dataset.metadata.to_internal_json_structure(), dataset_copy.metadata.to_internal_json_structure(), name)
            self.assertEqual(dataset_copy.metadata.query(()).get('test'), 'foobar', name)

    def test_list_ndarray_int(self):
        # With custom metadata which should be preserved.
        l = container.List([1, 2, 3], {
            'foo': 'bar',
        }, generate_metadata=True)

        self.assertEqual(utils.to_json_structure(l.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }])

        array = container.ndarray(l, generate_metadata=True)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
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

        l2 = container.List(array, generate_metadata=True)

        self.assertEqual(utils.to_json_structure(l2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
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

    def test_dataframe_ndarray_int_noncompact_metadata(self):
        # With custom metadata which should be preserved.
        df = container.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, {
            'foo': 'bar',
        }, generate_metadata=False)

        df.metadata = df.metadata.generate(df, compact=False)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
                'structural_type': 'numpy.int64',
            },
        }])

        array = container.ndarray(df, generate_metadata=False)

        array.metadata = array.metadata.generate(array, compact=False)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
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
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

        df2 = container.DataFrame(array, generate_metadata=False)

        df2.metadata = df2.metadata.generate(df2, compact=False)

        self.assertEqual(utils.to_json_structure(df2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
                'structural_type': 'numpy.int64',
            },
        }])

    def test_dataframe_ndarray_int_compact_metadata(self):
        # With custom metadata which should be preserved.
        df = container.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, {
            'foo': 'bar',
        }, generate_metadata=False)

        df.metadata = df.metadata.generate(df, compact=True)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

        array = container.ndarray(df, generate_metadata=False)

        array.metadata = array.metadata.generate(array, compact=True)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
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
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

        df2 = container.DataFrame(array, generate_metadata=False)

        df2.metadata = df2.metadata.generate(df2, compact=True)

        self.assertEqual(utils.to_json_structure(df2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

    def test_dataframe_list_int_compact_metadata(self):
        # With custom metadata which should be preserved.
        df = container.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, {
            'foo': 'bar',
        }, generate_metadata=False)

        df.metadata = df.metadata.generate(df, compact=True)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

        l = container.List(df, generate_metadata=False)

        l.metadata = l.metadata.generate(l, compact=True)

        self.assertEqual(utils.to_json_structure(l.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
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
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

        df2 = container.DataFrame(l, generate_metadata=False)

        df2.metadata = df2.metadata.generate(df2, compact=True)

        self.assertEqual(utils.to_json_structure(df2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
                # This is not really required, but current implementation adds it.
                # It is OK if in the future this gets removed.
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

    def test_dataframe_list_int_noncompact_metadata(self):
        # With custom metadata which should be preserved.
        df = container.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}, {
            'foo': 'bar',
        }, generate_metadata=False)

        df.metadata = df.metadata.generate(df, compact=False)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
                'structural_type': 'numpy.int64',
            },
        }])

        l = container.List(df, generate_metadata=False)

        l.metadata = l.metadata.generate(l, compact=False)

        self.assertEqual(utils.to_json_structure(l.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.list.List',
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
                'structural_type': 'd3m.container.list.List',
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
            },
        }])

        df2 = container.DataFrame(l, generate_metadata=False)

        df2.metadata = df2.metadata.generate(df2, compact=False)

        self.assertEqual(utils.to_json_structure(df2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
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
                # This is not really required, but current implementation adds it.
                # It is OK if in the future this gets removed.
                'structural_type': '__NO_VALUE__',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'int',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'A',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'B',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'C',
                'structural_type': 'numpy.int64',
            },
        }])

    def test_deep_ndarray_compact_metadata(self):
        # With custom metadata which should be preserved.
        array = container.ndarray(numpy.arange(3 * 4 * 5 * 5 * 5).reshape((3, 4, 5, 5, 5)), {
            'foo': 'bar',
        }, generate_metadata=False)
        array.metadata = array.metadata.generate(array, compact=True)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

        df = container.DataFrame(array, generate_metadata=False)
        df.metadata = df.metadata.generate(df, compact=True)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

        array2 = container.ndarray(df, generate_metadata=False)
        array2.metadata = array2.metadata.generate(array2, compact=True)

        # We do not automatically compact numpy with nested numpy arrays into one array
        # (there might be an exception if array is jagged).
        self.assertEqual(utils.to_json_structure(array2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'foo': 'bar',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

    def test_deep_ndarray_noncompact_metadata(self):
        # With custom metadata which should be preserved.
        array = container.ndarray(numpy.arange(3 * 4 * 5 * 5 * 5).reshape((3, 4, 5, 5, 5)), {
            'foo': 'bar',
        }, generate_metadata=False)
        array.metadata = array.metadata.generate(array, compact=False)

        self.assertEqual(utils.to_json_structure(array.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

        df = container.DataFrame(array, generate_metadata=False)
        df.metadata = df.metadata.generate(df, compact=False)

        self.assertEqual(utils.to_json_structure(df.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'foo': 'bar',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0, '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1, '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2, '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3, '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3, '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

        array2 = container.ndarray(df, generate_metadata=False)
        array2.metadata = array2.metadata.generate(array2, compact=False)

        # We do not automatically compact numpy with nested numpy arrays into one array
        # (there might be an exception if array is jagged).
        self.assertEqual(utils.to_json_structure(array2.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'foo': 'bar',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 4,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
                'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 5,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

    def test_simple_list_to_dataframe(self):
        data = container.List([1, 2, 3], generate_metadata=True)

        dataframe = container.DataFrame(data, generate_metadata=False)

        compact_metadata = dataframe.metadata.generate(dataframe, compact=True)
        noncompact_metadata = dataframe.metadata.generate(dataframe, compact=False)

        expected_metadata = [{
            'selector': [],
            'metadata': {
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'structural_type': 'd3m.container.pandas.DataFrame',
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': '__NO_VALUE__',
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }]

        self.assertEqual(utils.to_json_structure(compact_metadata.to_internal_simple_structure()), expected_metadata)

        expected_metadata[2]['selector'] = ['__ALL_ELEMENTS__', 0]

        self.assertEqual(utils.to_json_structure(noncompact_metadata.to_internal_simple_structure()), expected_metadata)

    def test_select_columns_compact_metadata(self):
        data = container.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}, generate_metadata=False)

        data.metadata = data.metadata.generate(data, compact=True)

        data.metadata = data.metadata.update_column(0, {'name': 'aaa'})
        data.metadata = data.metadata.update_column(1, {'name': 'bbb'})
        data.metadata = data.metadata.update_column(2, {'name': 'ccc'})
        data.metadata = data.metadata.update((0, 0), {'row': '1'})
        data.metadata = data.metadata.update((1, 0), {'row': '2'})
        data.metadata = data.metadata.update((2, 0), {'row': '3'})
        data.metadata = data.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowA'})

        data_metadata_before = data.metadata.to_internal_json_structure()

        # Test "select_columns" working with a tuple. Specifically, iloc[:, tuple(1)] does not work
        # (i.e. throws "{IndexingError}Too many indexers"), but iloc[:, 1] and iloc[:, [1]] work.
        selected = data.select_columns(tuple([1, 0, 2, 1]))

        self.assertEqual(selected.values.tolist(), [[4, 1, 7, 4], [5, 2, 8, 5], [6, 3, 9, 6]])

        self.assertEqual(utils.to_json_structure(selected.metadata.to_internal_simple_structure()), [{
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
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'bbb'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'aaa'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'ccc'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'bbb'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 1],
            'metadata': {'row': '1'},
        }, {
            'selector': [1, 1],
            'metadata': {'row': '2'},
        }, {
            'selector': [2, 1],
            'metadata': {'row': '3'},
        }])

        self.assertEqual(data.metadata.to_internal_json_structure(), data_metadata_before)

        selected = data.select_columns([1])

        self.assertEqual(selected.values.tolist(), [[4], [5], [6]])

        self.assertEqual(utils.to_json_structure(selected.metadata.to_internal_simple_structure()), [{
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
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'bbb'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }])

        self.assertEqual(data.metadata.to_internal_json_structure(), data_metadata_before)

    def test_select_columns_noncompact_metadata(self):
        data = container.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}, generate_metadata=False)

        data.metadata = data.metadata.generate(data, compact=False)

        data.metadata = data.metadata.update_column(0, {'name': 'aaa'})
        data.metadata = data.metadata.update_column(1, {'name': 'bbb'})
        data.metadata = data.metadata.update_column(2, {'name': 'ccc'})
        data.metadata = data.metadata.update((0, 0), {'row': '1'})
        data.metadata = data.metadata.update((1, 0), {'row': '2'})
        data.metadata = data.metadata.update((2, 0), {'row': '3'})
        data.metadata = data.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowA'})

        data_metadata_before = data.metadata.to_internal_json_structure()

        # Test "select_columns" working with a tuple. Specifically, iloc[:, tuple(1)] does not work
        # (i.e. throws "{IndexingError}Too many indexers"), but iloc[:, 1] and iloc[:, [1]] work.
        selected = data.select_columns(tuple([1, 0, 2, 1]))

        self.assertEqual(selected.values.tolist(), [[4, 1, 7, 4], [5, 2, 8, 5], [6, 3, 9, 6]])

        self.assertEqual(utils.to_json_structure(selected.metadata.to_internal_simple_structure()), [{
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
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'bbb', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'aaa', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'ccc', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'bbb', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 1],
            'metadata': {'row': '1'},
        }, {
            'selector': [1, 1],
            'metadata': {'row': '2'},
        }, {
            'selector': [2, 1],
            'metadata': {'row': '3'},
        }])

        self.assertEqual(data.metadata.to_internal_json_structure(), data_metadata_before)

        selected = data.select_columns([1])

        self.assertEqual(selected.values.tolist(), [[4], [5], [6]])

        self.assertEqual(utils.to_json_structure(selected.metadata.to_internal_simple_structure()), [{
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
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'bbb', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }])

        self.assertEqual(data.metadata.to_internal_json_structure(), data_metadata_before)

    def test_append_columns_compact_metadata(self):
        left = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9]}, {
            'top_level': 'left',
        }, generate_metadata=False)
        left.metadata = left.metadata.generate(left, compact=True)

        left.metadata = left.metadata.update_column(0, {'name': 'aaa111'})
        left.metadata = left.metadata.update_column(1, {'name': 'bbb111'})
        left.metadata = left.metadata.update_column(2, {'name': 'ccc111'})
        left.metadata = left.metadata.update((0, 0), {'row': '1a'})
        left.metadata = left.metadata.update((1, 0), {'row': '2a'})
        left.metadata = left.metadata.update((2, 0), {'row': '3a'})
        left.metadata = left.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowA'})

        right = container.DataFrame({'a2': [11, 12, 13], 'b2': [14, 15, 16], 'c2': [17, 18, 19]}, {
            'top_level': 'right',
        }, generate_metadata=False)
        right.metadata = right.metadata.generate(right, compact=True)

        right.metadata = right.metadata.update_column(0, {'name': 'aaa222'})
        right.metadata = right.metadata.update_column(1, {'name': 'bbb222'})
        right.metadata = right.metadata.update_column(2, {'name': 'ccc222'})
        right.metadata = right.metadata.update((0, 1), {'row': '1b'})
        right.metadata = right.metadata.update((1, 1), {'row': '2b'})
        right.metadata = right.metadata.update((2, 1), {'row': '3b'})
        right.metadata = right.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowB'})

        right_metadata_before = right.metadata.to_internal_json_structure()

        data = left.append_columns(right, use_right_metadata=False)

        self.assertEqual(data.values.tolist(), [[1, 4, 7, 11, 14, 17], [2, 5, 8, 12, 15, 18], [3, 6, 9, 13, 16, 19]])

        self.assertEqual(utils.to_json_structure(data.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'left',
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
            'metadata': {'name': 'aaa111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'bbb111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'ccc111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {'name': 'ccc222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a'},
        }, {
            'selector': [0, 3],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 4],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 5],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 4],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 4],
            'metadata': {'row': '3b'},
        }])

        data = left.append_columns(right, use_right_metadata=True)

        self.assertEqual(data.values.tolist(), [[1, 4, 7, 11, 14, 17], [2, 5, 8, 12, 15, 18], [3, 6, 9, 13, 16, 19]])

        self.assertEqual(utils.to_json_structure(data.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'right',
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
            'metadata': {'name': 'aaa111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'bbb111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'ccc111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'aaa222'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'name': 'bbb222'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {'name': 'ccc222'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a', 'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 1],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 2],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 4],
            'metadata': {'row': '1b'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 4],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 4],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(right.metadata.to_internal_json_structure(), right_metadata_before)

    def test_append_columns_noncompact_metadata(self):
        left = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9]}, {
            'top_level': 'left',
        }, generate_metadata=False)
        left.metadata = left.metadata.generate(left, compact=False)

        left.metadata = left.metadata.update_column(0, {'name': 'aaa111'})
        left.metadata = left.metadata.update_column(1, {'name': 'bbb111'})
        left.metadata = left.metadata.update_column(2, {'name': 'ccc111'})
        left.metadata = left.metadata.update((0, 0), {'row': '1a'})
        left.metadata = left.metadata.update((1, 0), {'row': '2a'})
        left.metadata = left.metadata.update((2, 0), {'row': '3a'})
        left.metadata = left.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowA'})

        right = container.DataFrame({'a2': [11, 12, 13], 'b2': [14, 15, 16], 'c2': [17, 18, 19]}, {
            'top_level': 'right',
        }, generate_metadata=False)
        right.metadata = right.metadata.generate(right, compact=False)

        right.metadata = right.metadata.update_column(0, {'name': 'aaa222'})
        right.metadata = right.metadata.update_column(1, {'name': 'bbb222'})
        right.metadata = right.metadata.update_column(2, {'name': 'ccc222'})
        right.metadata = right.metadata.update((0, 1), {'row': '1b'})
        right.metadata = right.metadata.update((1, 1), {'row': '2b'})
        right.metadata = right.metadata.update((2, 1), {'row': '3b'})
        right.metadata = right.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowB'})

        right_metadata_before = right.metadata.to_internal_json_structure()

        data = left.append_columns(right, use_right_metadata=False)

        self.assertEqual(data.values.tolist(), [[1, 4, 7, 11, 14, 17], [2, 5, 8, 12, 15, 18], [3, 6, 9, 13, 16, 19]])

        self.assertEqual(utils.to_json_structure(data.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'left',
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
            'metadata': {'name': 'aaa111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'bbb111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'ccc111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {'name': 'ccc222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a'},
        }, {
            'selector': [0, 3],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 4],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 5],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 4],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 4],
            'metadata': {'row': '3b'},
        }])

        data = left.append_columns(right, use_right_metadata=True)

        self.assertEqual(data.values.tolist(), [[1, 4, 7, 11, 14, 17], [2, 5, 8, 12, 15, 18], [3, 6, 9, 13, 16, 19]])

        self.assertEqual(utils.to_json_structure(data.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'top_level': 'right',
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
            'metadata': {'name': 'aaa111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'bbb111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'ccc111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {'name': 'ccc222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a', 'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 1],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 2],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 4],
            'metadata': {'row': '1b'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 4],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 4],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(right.metadata.to_internal_json_structure(), right_metadata_before)

    def test_replace_columns_compact_metadata(self):
        main = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9]}, {
            'top_level': 'main',
        }, generate_metadata=False)
        main.metadata = main.metadata.generate(main, compact=True)

        main.metadata = main.metadata.update_column(0, {'name': 'aaa111'})
        main.metadata = main.metadata.update_column(1, {'name': 'bbb111', 'extra': 'b_column'})
        main.metadata = main.metadata.update_column(2, {'name': 'ccc111'})
        main.metadata = main.metadata.update((0, 0), {'row': '1a'})
        main.metadata = main.metadata.update((1, 0), {'row': '2a'})
        main.metadata = main.metadata.update((2, 0), {'row': '3a'})
        main.metadata = main.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowA'})

        main_metadata_before = main.metadata.to_internal_json_structure()

        columns = container.DataFrame({'a2': [11, 12, 13], 'b2': [14, 15, 16]}, {
            'top_level': 'columns',
        }, generate_metadata=False)
        columns.metadata = columns.metadata.generate(columns, compact=True)

        columns.metadata = columns.metadata.update_column(0, {'name': 'aaa222'})
        columns.metadata = columns.metadata.update_column(1, {'name': 'bbb222'})
        columns.metadata = columns.metadata.update((0, 1), {'row': '1b'})
        columns.metadata = columns.metadata.update((1, 1), {'row': '2b'})
        columns.metadata = columns.metadata.update((2, 1), {'row': '3b'})
        columns.metadata = columns.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowB'})

        columns_metadata_before = columns.metadata.to_internal_json_structure()

        new_main = main.replace_columns(columns, [1, 2])

        self.assertEqual(new_main.values.tolist(), [[1, 11, 14], [2, 12, 15], [3, 13, 16]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'aaa111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a'},
        }, {
            'selector': [0, 1],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 2],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 2],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 2],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

        new_main = main.replace_columns(columns, [0, 2])

        self.assertEqual(new_main.values.tolist(), [[11, 4, 14], [12, 5, 15], [13, 6, 16]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 2],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 2],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 2],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

        new_main = main.replace_columns(columns, [1])

        self.assertEqual(new_main.values.tolist(), [[1, 11, 14, 7], [2, 12, 15, 8], [3, 13, 16, 9]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'metadata': {'name': 'aaa111'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'ccc111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a'},
        }, {
            'selector': [0, 1],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 2],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 3],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 2],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 2],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

        new_main = main.replace_columns(columns, [0, 1, 2])

        self.assertEqual(new_main.values.tolist(), [[11, 14], [12, 15], [13, 16]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 1],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 1],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 1],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

    def test_replace_columns_noncompact_metadata(self):
        main = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6], 'c1': [7, 8, 9]}, {
            'top_level': 'main',
        }, generate_metadata=False)
        main.metadata = main.metadata.generate(main, compact=False)

        main.metadata = main.metadata.update_column(0, {'name': 'aaa111'})
        main.metadata = main.metadata.update_column(1, {'name': 'bbb111', 'extra': 'b_column'})
        main.metadata = main.metadata.update_column(2, {'name': 'ccc111'})
        main.metadata = main.metadata.update((0, 0), {'row': '1a'})
        main.metadata = main.metadata.update((1, 0), {'row': '2a'})
        main.metadata = main.metadata.update((2, 0), {'row': '3a'})
        main.metadata = main.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowA'})

        main_metadata_before = main.metadata.to_internal_json_structure()

        columns = container.DataFrame({'a2': [11, 12, 13], 'b2': [14, 15, 16]}, {
            'top_level': 'columns',
        }, generate_metadata=False)
        columns.metadata = columns.metadata.generate(columns, compact=False)

        columns.metadata = columns.metadata.update_column(0, {'name': 'aaa222'})
        columns.metadata = columns.metadata.update_column(1, {'name': 'bbb222'})
        columns.metadata = columns.metadata.update((0, 1), {'row': '1b'})
        columns.metadata = columns.metadata.update((1, 1), {'row': '2b'})
        columns.metadata = columns.metadata.update((2, 1), {'row': '3b'})
        columns.metadata = columns.metadata.update((0, metadata_base.ALL_ELEMENTS), {'all_elements_on_row': 'rowB'})

        columns_metadata_before = columns.metadata.to_internal_json_structure()

        new_main = main.replace_columns(columns, [1, 2])

        self.assertEqual(new_main.values.tolist(), [[1, 11, 14], [2, 12, 15], [3, 13, 16]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'metadata': {'name': 'aaa111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a'},
        }, {
            'selector': [0, 1],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 2],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 2],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 2],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

        new_main = main.replace_columns(columns, [0, 2])

        self.assertEqual(new_main.values.tolist(), [[11, 4, 14], [12, 5, 15], [13, 6, 16]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'bbb111',
                'extra': 'b_column',
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 2],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 2],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 2],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

        new_main = main.replace_columns(columns, [1])

        self.assertEqual(new_main.values.tolist(), [[1, 11, 14, 7], [2, 12, 15, 8], [3, 13, 16, 9]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
            'metadata': {'name': 'aaa111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {'name': 'ccc111', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'row': '1a'},
        }, {
            'selector': [0, 1],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 2],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 3],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [1, 0],
            'metadata': {'row': '2a'},
        }, {
            'selector': [1, 2],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 0],
            'metadata': {'row': '3a'},
        }, {
            'selector': [2, 2],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

        new_main = main.replace_columns(columns, [0, 1, 2])

        self.assertEqual(new_main.values.tolist(), [[11, 14], [12, 15], [13, 16]])

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [], 'metadata': {
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
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'aaa222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'bbb222', 'structural_type': 'numpy.int64'},
        }, {
            'selector': [0, '__ALL_ELEMENTS__'],
            'metadata': {'all_elements_on_row': 'rowA'},
        }, {
            'selector': [0, 0],
            'metadata': {'all_elements_on_row': 'rowB'},
        }, {
            'selector': [0, 1],
            'metadata': {'row': '1b', 'all_elements_on_row': 'rowB'},
        }, {
            'selector': [1, 1],
            'metadata': {'row': '2b'},
        }, {
            'selector': [2, 1],
            'metadata': {'row': '3b'},
        }])

        self.assertEqual(main_metadata_before, main.metadata.to_internal_json_structure())
        self.assertEqual(columns_metadata_before, columns.metadata.to_internal_json_structure())

    def test_select_columns_empty(self):
        data = container.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}, generate_metadata=True)

        with self.assertRaises(Exception):
            data.select_columns([])

        with self.assertRaises(Exception):
            data.metadata.select_columns([])

        selected = data.select_columns([], allow_empty_columns=True)

        self.assertEqual(selected.shape, (3, 0))

        self.assertEqual(utils.to_json_structure(selected.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 3,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
               'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
               'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
               'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 0,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }])

    def test_dataframe_select_copy(self):
        df = container.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        selection = df.select_columns([0])

        with warnings.catch_warnings(record=True) as w:
            selection.iloc[:, 0] = selection.iloc[:, 0].map(lambda x: x + 1)

        self.assertEqual(len(w), 0)

        self.assertEqual(selection.values.tolist(), [[2], [3], [4]])
        self.assertEqual(df.values.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_save_container_empty_dataset(self):
        dataset = container.Dataset({}, generate_metadata=True)

        with tempfile.TemporaryDirectory() as temp_directory:
            container_utils.save_container(dataset, os.path.join(temp_directory, 'dataset'))


if __name__ == '__main__':
    unittest.main()
