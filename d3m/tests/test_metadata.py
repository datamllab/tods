import unittest

import jsonschema
import numpy

from d3m import container, utils
from d3m.metadata import base


def copy_elements_metadata(source_metadata, target_metadata, from_selector, to_selector=(), *, ignore_all_elements=False):
    return source_metadata._copy_elements_metadata(target_metadata, list(from_selector), list(to_selector), [], ignore_all_elements)


class TestMetadata(unittest.TestCase):
    def test_basic(self):
        md1 = base.Metadata({'value': 'test'})

        self.assertEqual(md1.query(()), {'value': 'test'})
        self.assertEqual(md1.query(('foo',)), {})
        self.assertEqual(md1.query(('bar',)), {})

        md2 = md1.update((), {'value2': 'test2'})

        self.assertEqual(md1.query(()), {'value': 'test'})
        self.assertEqual(md1.query(('foo',)), {})
        self.assertEqual(md1.query(('bar',)), {})
        self.assertEqual(md2.query(()), {'value': 'test', 'value2': 'test2'})

        md3 = md2.update(('foo',), {'element': 'one'})

        self.assertEqual(md1.query(()), {'value': 'test'})
        self.assertEqual(md1.query(('foo',)), {})
        self.assertEqual(md1.query(('bar',)), {})
        self.assertEqual(md2.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md3.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md3.query(('foo',)), {'element': 'one'})

        md4 = md3.update((base.ALL_ELEMENTS,), {'element': 'two'})

        self.assertEqual(md1.query(()), {'value': 'test'})
        self.assertEqual(md1.query(('foo',)), {})
        self.assertEqual(md1.query(('bar',)), {})
        self.assertEqual(md2.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md3.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md3.query(('foo',)), {'element': 'one'})
        self.assertEqual(md4.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md4.query((base.ALL_ELEMENTS,)), {'element': 'two'})
        self.assertEqual(md4.query(('foo',)), {'element': 'two'})

        md5 = md4.update(('foo',), {'element': 'three'})

        self.assertEqual(md1.query(()), {'value': 'test'})
        self.assertEqual(md1.query(('foo',)), {})
        self.assertEqual(md1.query(('bar',)), {})
        self.assertEqual(md2.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md3.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md3.query(('foo',)), {'element': 'one'})
        self.assertEqual(md4.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md4.query((base.ALL_ELEMENTS,)), {'element': 'two'})
        self.assertEqual(md4.query(('foo',)), {'element': 'two'})
        self.assertEqual(md5.query(()), {'value': 'test', 'value2': 'test2'})
        self.assertEqual(md5.query((base.ALL_ELEMENTS,)), {'element': 'two'})
        self.assertEqual(md5.query(('foo',)), {'element': 'three'})

    def test_all_elements(self):
        md1 = base.Metadata()

        md2 = md1.update((base.ALL_ELEMENTS, 'bar'), {'value': 'test1'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})

        md3 = md2.update(('foo', 'bar'), {'value': 'test2'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})
        self.assertEqual(md3.query(('foo', 'bar')), {'value': 'test2'})

        md4 = md3.update((base.ALL_ELEMENTS, 'bar'), {'value': 'test3'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})
        self.assertEqual(md3.query(('foo', 'bar')), {'value': 'test2'})
        self.assertEqual(md4.query(('foo', 'bar')), {'value': 'test3'})

        md5 = md4.update(('foo', base.ALL_ELEMENTS), {'value': 'test4'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})
        self.assertEqual(md3.query(('foo', 'bar')), {'value': 'test2'})
        self.assertEqual(md4.query(('foo', 'bar')), {'value': 'test3'})
        self.assertEqual(md5.query(('foo', 'bar')), {'value': 'test4'})

        md6 = md5.update(('foo', 'bar'), {'value': 'test5'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})
        self.assertEqual(md3.query(('foo', 'bar')), {'value': 'test2'})
        self.assertEqual(md4.query(('foo', 'bar')), {'value': 'test3'})
        self.assertEqual(md5.query(('foo', 'bar')), {'value': 'test4'})
        self.assertEqual(md6.query(('foo', 'bar')), {'value': 'test5'})

        md7 = md6.update((base.ALL_ELEMENTS, base.ALL_ELEMENTS), {'value': 'test6'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})
        self.assertEqual(md3.query(('foo', 'bar')), {'value': 'test2'})
        self.assertEqual(md4.query(('foo', 'bar')), {'value': 'test3'})
        self.assertEqual(md5.query(('foo', 'bar')), {'value': 'test4'})
        self.assertEqual(md6.query(('foo', 'bar')), {'value': 'test5'})
        self.assertEqual(md7.query(('foo', 'bar')), {'value': 'test6'})

        md8 = md7.update(('foo', 'bar'), {'value': 'test7'})

        self.assertEqual(md2.query(('foo', 'bar')), {'value': 'test1'})
        self.assertEqual(md3.query(('foo', 'bar')), {'value': 'test2'})
        self.assertEqual(md4.query(('foo', 'bar')), {'value': 'test3'})
        self.assertEqual(md5.query(('foo', 'bar')), {'value': 'test4'})
        self.assertEqual(md6.query(('foo', 'bar')), {'value': 'test5'})
        self.assertEqual(md7.query(('foo', 'bar')), {'value': 'test6'})
        self.assertEqual(md8.query(('foo', 'bar')), {'value': 'test7'})

        self.assertEqual(md8.to_internal_json_structure(), [{
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'value': 'test6'
            }
        }, {
            'selector': ['foo', 'bar'],
            'metadata': {
                'value': 'test7'
            }
        }])

    def test_removal(self):
        md1 = base.Metadata().update((), {'value': 'test1'})

        self.assertEqual(md1.query(()), {'value': 'test1'})

        md2 = md1.update((), {'value': base.NO_VALUE})

        self.assertEqual(md1.query(()), {'value': 'test1'})
        self.assertEqual(md2.query(()), {})
        self.assertEqual(md2.query((), ignore_all_elements=True), {})

        md3 = md2.update((), {'value': {'value2': 'test2'}})

        self.assertEqual(md1.query(()), {'value': 'test1'})
        self.assertEqual(md2.query(()), {})
        self.assertEqual(md3.query(()), {'value': {'value2': 'test2'}})

        md4 = md3.update((), {'value': {'value2': base.NO_VALUE}})

        self.assertEqual(md1.query(()), {'value': 'test1'})
        self.assertEqual(md2.query(()), {})
        self.assertEqual(md3.query(()), {'value': {'value2': 'test2'}})
        self.assertEqual(md4.query(()), {})

        md5 = md4.update((), {'value': base.NO_VALUE})

        self.assertEqual(md1.query(()), {'value': 'test1'})
        self.assertEqual(md2.query(()), {})
        self.assertEqual(md3.query(()), {'value': {'value2': 'test2'}})
        self.assertEqual(md4.query(()), {})
        self.assertEqual(md5.query(()), {})

    def test_empty_dict(self):
        md = base.Metadata().update((), {'value': {}})

        self.assertEqual(md.query(()), {'value': {}})

        md = md.update((), {'value': {'a': '1', 'b': 2}})

        self.assertEqual(md.query(()), {'value': {'a': '1', 'b': 2}})

        md = md.update((), {'value': {'a': base.NO_VALUE, 'b': base.NO_VALUE}})

        self.assertEqual(md.query(()), {})

        md = md.update((), {'value': {'a': '1', 'b': 2}})

        self.assertEqual(md.query(()), {'value': {'a': '1', 'b': 2}})

        md = md.update((), {'value': {'a': base.NO_VALUE}})

        self.assertEqual(md.query(()), {'value': {'b': 2}})

    def test_remove(self):
        metadata = base.Metadata().update((), {'value': 'test1'})
        metadata = metadata.update(('a',), {'value': 'test2'})
        metadata = metadata.update(('a', 'b'), {'value': 'test3'})
        metadata = metadata.update(('a', 'b', 'c'), {'value': 'test4'})
        metadata = metadata.update((base.ALL_ELEMENTS, 'b', 'd'), {'value': 'test5'})
        metadata = metadata.update((base.ALL_ELEMENTS, 'b', 'e', base.ALL_ELEMENTS), {'value': 'test6'})

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b'],
                'metadata': {
                    'value': 'test3',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove(('a', 'b'))

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove(('a', 'b'), recursive=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
        ])

        new_metadata = metadata.remove((), recursive=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [])

        new_metadata = metadata.remove((base.ALL_ELEMENTS, 'b'))

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove((base.ALL_ELEMENTS, 'b'), recursive=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
        ])

        new_metadata = metadata.remove((base.ALL_ELEMENTS, 'b'), strict_all_elements=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b'],
                'metadata': {
                    'value': 'test3',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove((base.ALL_ELEMENTS, 'b'), recursive=True, strict_all_elements=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b'],
                'metadata': {
                    'value': 'test3',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove(('a', base.ALL_ELEMENTS))

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove(('a', base.ALL_ELEMENTS), strict_all_elements=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b'],
                'metadata': {
                    'value': 'test3',
                },
            },
            {
                'selector': ['a', 'b', 'c'],
                'metadata': {
                    'value': 'test4',
                },
            },
        ])

        new_metadata = metadata.remove(('a', base.ALL_ELEMENTS), recursive=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'd'],
                'metadata': {
                    'value': 'test5',
                },
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
        ])

        new_metadata = metadata.remove((base.ALL_ELEMENTS, 'b', base.ALL_ELEMENTS))

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['__ALL_ELEMENTS__', 'b', 'e', '__ALL_ELEMENTS__'],
                'metadata': {
                    'value': 'test6',
                },
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b'],
                'metadata': {
                    'value': 'test3',
                },
            },
        ])

        new_metadata = metadata.remove((base.ALL_ELEMENTS, 'b', base.ALL_ELEMENTS), recursive=True)

        self.assertEqual(utils.to_json_structure(new_metadata.to_internal_simple_structure()), [
            {
                'selector': [],
                'metadata': {
                    'value': 'test1',
                }
            },
            {
                'selector': ['a'],
                'metadata': {
                    'value': 'test2',
                },
            },
            {
                'selector': ['a', 'b'],
                'metadata': {
                    'value': 'test3',
                },
            },
        ])

    def test_remove_column(self):
        metadata = base.DataMetadata().update((base.ALL_ELEMENTS, 0), {'name': 'column1'})
        metadata = metadata.update((base.ALL_ELEMENTS, 1), {'name': 'column2'})
        metadata = metadata.update((10, 0), {'value': 'row10.0'})
        metadata = metadata.update((10, 1), {'value': 'row10.1'})

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'name': 'column1'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'column2'},
        }, {
            'selector': [10, 0],
            'metadata': {'value': 'row10.0'},
        }, {
            'selector': [10, 1],
            'metadata': {'value': 'row10.1'},
        }])

        metadata = metadata.remove_column(0)

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'name': 'column2'},
        }, {
            'selector': [10, 1],
            'metadata': {'value': 'row10.1'},
        }])

    def test_check(self):
        data = container.Dataset({
            '0': container.ndarray(numpy.array([
                [1, 2, 3],
                [4, 5, 6],
            ])),
        })

        md1 = base.DataMetadata().update((), {
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': type(data),
            'value': 'test'
        })

        md1.check(data)

        md2 = md1.update(('missing',), {'value': 'test'})

        with self.assertRaisesRegex(ValueError, 'cannot be resolved'):
            md2.check(data)

        md3 = md1.update(('0', 1), {'value': 'test'})

        md4 = md3.update(('0', 2), {'value': 'test'})

        with self.assertRaisesRegex(ValueError, 'cannot be resolved'):
            md4.check(data)

        md5 = md3.update(('0', 1, 3), {'value': 'test'})

        with self.assertRaisesRegex(ValueError, 'cannot be resolved'):
            md5.check(data)

        md6 = md3.update(('0', 1, 2, base.ALL_ELEMENTS), {'value': 'test'})

        with self.assertRaisesRegex(ValueError, 'ALL_ELEMENTS set but dimension missing at'):
            md6.check(data)

    def test_errors(self):
        with self.assertRaisesRegex(TypeError, 'Metadata should be a dict'):
            base.Metadata().update((), None)

        class Custom:
            pass

        with self.assertRaisesRegex(TypeError, 'is not known to be immutable'):
            base.Metadata().update((), {'foo': Custom()})

        with self.assertRaisesRegex(TypeError, 'Selector is not a tuple or a list'):
            base.Metadata().update({}, {'value': 'test'})

        with self.assertRaisesRegex(TypeError, 'is not a str, int, or ALL_ELEMENTS'):
            base.Metadata().update((1.0,), {'value': 'test'})

        with self.assertRaisesRegex(TypeError, 'is not a str, int, or ALL_ELEMENTS'):
            base.Metadata().update((None,), {'value': 'test'})

    def test_data(self):
        data = container.Dataset({
            '0': container.ndarray(numpy.array([
                [1, 2, 3],
                [4, 5, 6],
            ])),
        })

        md1 = base.DataMetadata()
        md1.update((), {'value': 'test'})

        with self.assertRaisesRegex(jsonschema.exceptions.ValidationError, 'is a required property'):
            md1.check(md1)

        md1 = base.DataMetadata().generate(data, compact=True)

        md2 = md1.update((), {
            'id': 'test-dataset',
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': type(data),
            'dimension': {
                'length': 1
            }
        })

        md3 = md2.update(('0',), {
            'structural_type': type(data['0']),
            'dimension': {
                'length': 2
            }
        })

        self.assertEqual(utils.to_json_structure(md3.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'id': 'test-dataset',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'dimension': {
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                    'length': 1
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 2,
                     'name': 'rows',
                     'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
               'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
               'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
           'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata':  {
                'dimension': {
                    'length': 3,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
           'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['0'],
            'metadata': {
                'structural_type': 'd3m.container.numpy.ndarray',
                'dimension': {
                    'length': 2,
                },
            },
        }])

        md1 = base.DataMetadata().generate(data, compact=False)

        md2 = md1.update((), {
            'id': 'test-dataset',
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': type(data),
            'dimension': {
                'length': 1
            }
        })

        md3 = md2.update(('0',), {
            'structural_type': type(data['0']),
            'dimension': {
                'length': 2
            }
        })

        self.assertEqual(utils.to_json_structure(md3.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'id': 'test-dataset',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'dimension': {
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                    'length': 1
                },
            },
        }, {
            'selector': ['0'],
            'metadata': {
                'dimension': {
                    'length': 2,
                     'name': 'rows',
                     'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
               'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
               'structural_type': 'd3m.container.numpy.ndarray',
            },
        }, {
           'selector': ['0', '__ALL_ELEMENTS__'],
            'metadata':  {
                'dimension': {
                    'length': 3,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
           'selector': ['0', '__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {
                'structural_type': 'numpy.int64',
            },
        }])

    def test_prune_bug(self):
        metadata = base.Metadata().update((base.ALL_ELEMENTS, 0), {'foo': 'bar1'})
        metadata = metadata.update((0, 1), {'foo': 'bar2'})
        metadata = metadata.update((1, 1), {'foo': 'bar2'})
        metadata = metadata.update((2, 1), {'foo': 'bar2'})
        metadata = metadata.update((base.ALL_ELEMENTS, base.ALL_ELEMENTS), {'foo': 'bar3'})

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'foo': 'bar3'},
        }])

    def test_remove_empty_metadata(self):
        metadata = base.Metadata().update((base.ALL_ELEMENTS,), {
            'foo': {
                'bar': 42,
            },
            'other': 1,
        })

        metadata = metadata.update((base.ALL_ELEMENTS,), {
            'foo': {
                'bar': base.NO_VALUE,
            },
        })

        self.assertEqual(metadata.query((base.ALL_ELEMENTS,)), {
            'other': 1,
        })

        metadata = base.Metadata({
            'foo': {
                'bar': 42,
            },
            'other': 1,
        })

        metadata = metadata.update((), {
            'foo': {
                'bar': base.NO_VALUE,
            },
        })

        self.assertEqual(metadata.query(()), {
            'other': 1,
        })

        metadata = base.Metadata({
            'foo': {
                'bar': 42,
            },
        })

        metadata = metadata.update((), {
            'foo': {
                'bar': base.NO_VALUE,
            },
        })

        self.assertEqual(metadata.query(()), {})

        metadata = base.Metadata().update(('a',), {
            'foo': {
                'bar': 42,
            },
        })

        metadata = metadata.update((base.ALL_ELEMENTS,), {
            'foo': {
                'bar': base.NO_VALUE,
            },
        })

        self.assertEqual(metadata.query(('a',)), {})

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'foo': {
                    'bar': '__NO_VALUE__',
                },
            },
        }])

    def test_ignore_all_elements(self):
        metadata = base.Metadata().update((base.ALL_ELEMENTS,), {
            'foo': 'bar',
            'other': 42,
        })

        metadata = metadata.update((0,), {
            'foo': base.NO_VALUE,
        })

        metadata = metadata.update((2,), {
            'other2': 43,
        })

        self.assertEqual(metadata.query((0,)), {'other': 42})
        self.assertEqual(metadata.query((1,)), {'foo': 'bar', 'other': 42})
        self.assertEqual(metadata.query((2,)), {'foo': 'bar', 'other': 42, 'other2': 43})
        self.assertEqual(metadata.query((0,), ignore_all_elements=True), {})
        self.assertEqual(metadata.query((1,), ignore_all_elements=True), {})
        self.assertEqual(metadata.query((2,), ignore_all_elements=True), {'other2': 43})

        metadata = metadata.update((base.ALL_ELEMENTS,), {
            'foo': 'bar2',
        })

        self.assertEqual(metadata.query((0,)), {'foo': 'bar2', 'other': 42})
        self.assertEqual(metadata.query((1,)), {'foo': 'bar2', 'other': 42})
        self.assertEqual(metadata.query((2,)), {'foo': 'bar2', 'other': 42, 'other2': 43})
        self.assertEqual(metadata.query((0,), ignore_all_elements=True), {})
        self.assertEqual(metadata.query((1,), ignore_all_elements=True), {})
        self.assertEqual(metadata.query((2,), ignore_all_elements=True), {'other2': 43})

    def test_query_with_exceptions(self):
        metadata = base.Metadata().update((base.ALL_ELEMENTS,), {
            'foo': 'bar',
            'other': 42,
        })

        metadata = metadata.update((0,), {
            'foo': base.NO_VALUE,
        })

        metadata = metadata.update((2,), {
            'other2': 43,
        })

        self.assertEqual(metadata.query((0,)), {'other': 42})
        self.assertEqual(metadata.query((1,)), {'foo': 'bar', 'other': 42})
        self.assertEqual(metadata.query((2,)), {'foo': 'bar', 'other': 42, 'other2': 43})

        self.assertEqual(metadata.query_with_exceptions((0,)), ({'other': 42}, {}))
        self.assertEqual(metadata.query_with_exceptions((1,)), ({'foo': 'bar', 'other': 42}, {}))
        self.assertEqual(metadata.query_with_exceptions((2,)), ({'foo': 'bar', 'other': 42, 'other2': 43}, {}))

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS,)), ({
            'foo': 'bar',
            'other': 42,
        }, {
            (0,): {'other': 42},
            (2,): {'foo': 'bar', 'other': 42, 'other2': 43},
        }))

        metadata = metadata.update((base.ALL_ELEMENTS,), {
            'foo': 'bar2',
        })

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS,)), ({
            'foo': 'bar2',
            'other': 42,
        }, {
            (2,): {'foo': 'bar2', 'other': 42, 'other2': 43},
        }))

        metadata = base.Metadata().update((base.ALL_ELEMENTS, 0), {
            'name': 'bar',
        })

        metadata = metadata.update((base.ALL_ELEMENTS, 1), {
            'name': 'foo',
        })

        metadata = metadata.update((2, 0), {
            'name': 'bar2',
        })

        metadata = metadata.update((2, 2), {
            'name': 'foo2',
        })

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, 0)), ({
            'name': 'bar',
        }, {
            (2, 0): {'name': 'bar2'},
        }))

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, 1)), ({
            'name': 'foo',
        }, {}))

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, 2)), ({}, {
            (2, 2): {'name': 'foo2'},
        }))

        self.assertEqual(metadata.query_with_exceptions((2, base.ALL_ELEMENTS)), ({}, {
            (2, 0): {'name': 'bar2'},
            (2, 2): {'name': 'foo2'},
        }))

        metadata = base.Metadata().update((base.ALL_ELEMENTS, base.ALL_ELEMENTS), {
            'foo': 'bar',
            'other': 42,
        })

        metadata = metadata.update((base.ALL_ELEMENTS, 0), {
            'foo': base.NO_VALUE,
        })

        metadata = metadata.update((base.ALL_ELEMENTS, 2), {
            'other2': 43,
        })

        self.assertEqual(metadata.query((base.ALL_ELEMENTS, 0)), {'other': 42})
        self.assertEqual(metadata.query((base.ALL_ELEMENTS, 1)), {'foo': 'bar', 'other': 42})
        self.assertEqual(metadata.query((base.ALL_ELEMENTS, 2)), {'foo': 'bar', 'other': 42, 'other2': 43})

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, 0)), ({'other': 42}, {}))
        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, 1)), ({'foo': 'bar', 'other': 42}, {}))
        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, 2)), ({'foo': 'bar', 'other': 42, 'other2': 43}, {}))

        self.assertEqual(metadata.query_with_exceptions((base.ALL_ELEMENTS, base.ALL_ELEMENTS)), ({
            'foo': 'bar',
            'other': 42,
        }, {
            (base.ALL_ELEMENTS, 0): {'other': 42},
            (base.ALL_ELEMENTS, 2): {'foo': 'bar', 'other': 42, 'other2': 43},
        }))

    def test_semantic_types(self):
        metadata = base.DataMetadata({
            'structural_type': container.DataFrame,
            'schema': base.CONTAINER_SCHEMA_VERSION,
        })

        self.assertFalse(metadata.has_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetResource'))

        metadata = metadata.add_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetResource')

        self.assertTrue(metadata.has_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetResource'))

        metadata = metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetResource')

        self.assertFalse(metadata.has_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetResource'))

        metadata = metadata.add_semantic_type((base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        metadata = metadata.add_semantic_type((base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        metadata = metadata.add_semantic_type((base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        self.assertEqual(metadata.get_elements_with_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/Attribute'), [])
        self.assertEqual(metadata.get_elements_with_semantic_type((base.ALL_ELEMENTS,), 'https://metadata.datadrivendiscovery.org/types/Attribute'), [0, 2])

    def test_copy_elements_metadata(self):
        metadata = base.Metadata()

        metadata = metadata.update((), {'level0': 'foobar0'})

        metadata = metadata.update(('level1',), {'level1': 'foobar1'})

        metadata = metadata.update((base.ALL_ELEMENTS,), {'level1a': 'foobar1a', 'level1b': 'foobar1b'})

        metadata = metadata.update(('level1',), {'level1b': base.NO_VALUE})

        metadata = metadata.update(('level1', 'level2'), {'level2': 'foobar2'})

        metadata = metadata.update((base.ALL_ELEMENTS, base.ALL_ELEMENTS), {'level2a': 'foobar2a', 'level2b': 'foobar2b'})

        metadata = metadata.update(('level1', 'level2'), {'level2b': base.NO_VALUE})

        metadata = metadata.update(('level1', 'level2', 'level3'), {'level3': 'foobar3'})

        metadata = metadata.update((base.ALL_ELEMENTS, base.ALL_ELEMENTS, 'level3'), {'level3a': 'foobar3a'})

        metadata = metadata.update(('level1', 'level2', 'level3.1'), {'level3.1': 'foobar3.1'})

        metadata = metadata.update(('level1', 'level2', 'level3', 'level4'), {'level4': 'foobar4'})

        metadata = metadata.update(('level1', 'level2', 'level3', 'level4.1'), {'level4.1': 'foobar4.1'})

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {'level0': 'foobar0'},
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level1a': 'foobar1a', 'level1b': 'foobar1b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level1'],
            'metadata': {'level1': 'foobar1', 'level1b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level1', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(metadata.query(('level1', 'level2')), {
            'level2a': 'foobar2a',
            'level2': 'foobar2',
        })

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})

        target_metadata = copy_elements_metadata(metadata, target_metadata, ())

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level1a': 'foobar1a', 'level1b': 'foobar1b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level1'],
            'metadata': {'level1': 'foobar1', 'level1b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level1', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(target_metadata.to_json_structure(), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level1a': 'foobar1a', 'level1b': 'foobar1b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level1'],
            'metadata': {'level1': 'foobar1', 'level1a': 'foobar1a'},
        }, {
            'selector': ['level1', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['level1', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level1', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2a': 'foobar2a'},
        }, {
            'selector': ['level1', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3', 'level3a': 'foobar3a'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level1', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})

        target_metadata = copy_elements_metadata(metadata, target_metadata, ('level1',))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(target_metadata.query(('level2',)), {
            'level2a': 'foobar2a',
            'level2': 'foobar2',
        })

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})
        target_metadata = target_metadata.update(('zlevel',), {'level1z': 'foobar1z'})

        target_metadata = copy_elements_metadata(metadata, target_metadata, ('level1',), ('zlevel',))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['zlevel'],
            'metadata': {'level1z': 'foobar1z'},
        }, {
            'selector': ['zlevel', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['zlevel', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['zlevel', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['zlevel', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['zlevel', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['zlevel', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['zlevel', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(target_metadata.query(('zlevel', 'level2',)), {
            'level2a': 'foobar2a',
            'level2': 'foobar2',
        })

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})

        target_metadata = copy_elements_metadata(metadata, target_metadata, ('level1', 'level2'))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['level3'],
            'metadata': {'level3': 'foobar3', 'level3a': 'foobar3a'},
        }, {
            'selector': ['level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})
        target_metadata = target_metadata.update(('zlevel',), {'level1z': 'foobar1z'})

        target_metadata = copy_elements_metadata(metadata, target_metadata, ('level1', 'level2'), ('zlevel',))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['zlevel'],
            'metadata': {'level1z': 'foobar1z'},
        }, {
            'selector': ['zlevel', 'level3'],
            'metadata': {'level3': 'foobar3', 'level3a': 'foobar3a'},
        }, {
            'selector': ['zlevel', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['zlevel', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['zlevel', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

    def test_copy_metadata(self):
        metadata = base.Metadata()

        metadata = metadata.update((), {'level0': 'foobar0'})

        metadata = metadata.update(('level1',), {'level1': 'foobar1'})

        metadata = metadata.update((base.ALL_ELEMENTS,), {'level1a': 'foobar1a', 'level1b': 'foobar1b'})

        metadata = metadata.update(('level1',), {'level1b': base.NO_VALUE})

        metadata = metadata.update(('level1', 'level2'), {'level2': 'foobar2'})

        metadata = metadata.update((base.ALL_ELEMENTS, base.ALL_ELEMENTS), {'level2a': 'foobar2a', 'level2b': 'foobar2b'})

        metadata = metadata.update(('level1', 'level2'), {'level2b': base.NO_VALUE})

        metadata = metadata.update(('level1', 'level2', 'level3'), {'level3': 'foobar3'})

        metadata = metadata.update((base.ALL_ELEMENTS, base.ALL_ELEMENTS, 'level3'), {'level3a': 'foobar3a'})

        metadata = metadata.update(('level1', 'level2', 'level3.1'), {'level3.1': 'foobar3.1'})

        metadata = metadata.update(('level1', 'level2', 'level3', 'level4'), {'level4': 'foobar4'})

        metadata = metadata.update(('level1', 'level2', 'level3', 'level4.1'), {'level4.1': 'foobar4.1'})

        self.assertEqual(utils.to_json_structure(metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {'level0': 'foobar0'},
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level1a': 'foobar1a', 'level1b': 'foobar1b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level1'],
            'metadata': {'level1': 'foobar1', 'level1b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level1', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(metadata.query(('level1', 'level2')), {
            'level2a': 'foobar2a',
            'level2': 'foobar2',
        })

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})

        target_metadata = metadata.copy_to(target_metadata, ())

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0': 'foobar0',
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level1a': 'foobar1a', 'level1b': 'foobar1b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level1'],
            'metadata': {'level1': 'foobar1', 'level1b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['level1', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level1', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level1', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})

        target_metadata = metadata.copy_to(target_metadata, ('level1',))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'level1': 'foobar1',
                'level1b': '__NO_VALUE__',
                'level1a': 'foobar1a',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(target_metadata.query(('level2',)), {
            'level2a': 'foobar2a',
            'level2': 'foobar2',
        })

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})
        target_metadata = target_metadata.update(('zlevel',), {'level1z': 'foobar1z'})

        target_metadata = metadata.copy_to(target_metadata, ('level1',), ('zlevel',))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['zlevel'],
            'metadata': {'level1z': 'foobar1z', 'level1': 'foobar1', 'level1b': '__NO_VALUE__', 'level1a': 'foobar1a'},
        }, {
            'selector': ['zlevel', '__ALL_ELEMENTS__'],
            'metadata': {'level2a': 'foobar2a', 'level2b': 'foobar2b'},
        }, {
            'selector': ['zlevel', '__ALL_ELEMENTS__', 'level3'],
            'metadata': {'level3a': 'foobar3a'},
        }, {
            'selector': ['zlevel', 'level2'],
            'metadata': {'level2': 'foobar2', 'level2b': '__NO_VALUE__'},
        }, {
            'selector': ['zlevel', 'level2', 'level3'],
            'metadata': {'level3': 'foobar3'},
        }, {
            'selector': ['zlevel', 'level2', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['zlevel', 'level2', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['zlevel', 'level2', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        self.assertEqual(target_metadata.query(('zlevel', 'level2',)), {
            'level2a': 'foobar2a',
            'level2': 'foobar2',
        })

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})

        target_metadata = metadata.copy_to(target_metadata, ('level1', 'level2'))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'level2': 'foobar2',
                'level2b': '__NO_VALUE__',
                'level2a': 'foobar2a',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['level3'],
            'metadata': {'level3': 'foobar3', 'level3a': 'foobar3a'},
        }, {
            'selector': ['level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

        target_metadata = base.DataMetadata({
            'schema': base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        })

        target_metadata = target_metadata.update((), {'level0z': 'foobar0z'})
        target_metadata = target_metadata.update(('zlevel',), {'level1z': 'foobar1z'})

        target_metadata = metadata.copy_to(target_metadata, ('level1', 'level2'), ('zlevel',))

        self.assertEqual(utils.to_json_structure(target_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'level0z': 'foobar0z',
                'schema': base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['zlevel'],
            'metadata': {'level1z': 'foobar1z', 'level2': 'foobar2', 'level2b': '__NO_VALUE__', 'level2a': 'foobar2a'},
        }, {
            'selector': ['zlevel', 'level3'],
            'metadata': {'level3': 'foobar3', 'level3a': 'foobar3a'},
        }, {
            'selector': ['zlevel', 'level3', 'level4'],
            'metadata': {'level4': 'foobar4'},
        }, {
            'selector': ['zlevel', 'level3', 'level4.1'],
            'metadata': {'level4.1': 'foobar4.1'},
        }, {
            'selector': ['zlevel', 'level3.1'],
            'metadata': {'level3.1': 'foobar3.1'},
        }])

    def test_get_index_columns(self):
        main = container.DataFrame({'a1': [1, 2, 3], 'b1': [4, 5, 6]}, generate_metadata=True)

        main.metadata = main.metadata.update((base.ALL_ELEMENTS, 0), {
            'name': 'image',
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
        })
        main.metadata = main.metadata.update((base.ALL_ELEMENTS, 1), {
            'name': 'd3mIndex',
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
        })

        self.assertEqual(main.metadata.get_index_columns(), [1, 0])

    def test_query_field(self):
        md = base.Metadata()
        md = md.update((1,), {'key': 'value'})

        self.assertEqual(md.query_field((1,), 'key', strict_all_elements=False), 'value')
        self.assertEqual(md.query_field((1,), 'key', strict_all_elements=True), 'value')

        with self.assertRaises(KeyError):
            self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=False), 'value')
        self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=True), 'value')

        with self.assertRaises(KeyError):
            self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key2', strict_all_elements=True), 'value')

        md = md.update((2,), {'key': 'value'})

        with self.assertRaises(KeyError):
            self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=False), 'value')
        self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=True), 'value')

        md = md.update((3,), {'key': 'value2'})

        with self.assertRaises(KeyError):
            self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=False), 'value')
        with self.assertRaises(KeyError):
            self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=True), 'value')

        md = md.update((base.ALL_ELEMENTS,), {'key': 'value'})

        self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=False), 'value')
        self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=True), 'value')

        self.assertEqual(md.query_field((1,), 'key', strict_all_elements=False), 'value')
        self.assertEqual(md.query_field((1,), 'key', strict_all_elements=True), 'value')

        md = md.update((3,), {'key': 'value2'})

        self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=False), 'value')
        with self.assertRaises(KeyError):
            self.assertEqual(md.query_field((base.ALL_ELEMENTS,), 'key', strict_all_elements=True), 'value')

    def test_query_field_with_exceptions(self):
        md = base.Metadata()
        md = md.update((1,), {'key': 'value'})
        md = md.update((2,), {'key': 'value2'})

        self.assertEqual(md.query_field_with_exceptions((1,), 'key'), ('value', {}))
        self.assertEqual(md.query_field_with_exceptions((2,), 'key'), ('value2', {}))
        with self.assertRaises(KeyError):
            md.query_field_with_exceptions((3,), 'key')

        self.assertEqual(md.query_field_with_exceptions((base.ALL_ELEMENTS,), 'key'), (base.NO_VALUE, {(1,): 'value', (2,): 'value2'}))

        # All elements ar require "key" field when there is no explicit ALL_ELEMENTS metadata.
        md = md.update((3,), {'key2': 'value'})

        with self.assertRaises(KeyError):
            md.query_field_with_exceptions((base.ALL_ELEMENTS,), 'key')

        md = md.update((base.ALL_ELEMENTS,), {'key': 'value'})

        self.assertEqual(md.query_field_with_exceptions((1,), 'key'), ('value', {}))
        self.assertEqual(md.query_field_with_exceptions((2,), 'key'), ('value', {}))

        self.assertEqual(md.query_field_with_exceptions((base.ALL_ELEMENTS,), 'key'), ('value', {}))

        md = md.update((3,), {'key': 'value2'})

        self.assertEqual(md.query_field_with_exceptions((base.ALL_ELEMENTS,), 'key'), ('value', {(3,): 'value2'}))

        # Setting same value as what ALL_ELEMENTS has should not add additional exception.
        md = md.update((4,), {'key': 'value'})

        self.assertEqual(md.query_field_with_exceptions((base.ALL_ELEMENTS,), 'key'), ('value', {(3,): 'value2'}))

        # Because ALL_ELEMENTS is set, any additional elements without "key" field are ignored.
        md = md.update((5,), {'key2': 'value'})

        self.assertEqual(md.query_field_with_exceptions((base.ALL_ELEMENTS,), 'key'), ('value', {(3,): 'value2'}))

    def test_compact_generated_metadata(self):
        ALL_GENERATED_KEYS = ['foo', 'name', 'other', 'structural_type']

        compacted_metadata = base.DataMetadata._compact_metadata({}, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {})

        # All equal.
        new_metadata = {
            ('a',): {'foo': 'bar', 'other': 1},
            ('b',): {'foo': 'bar', 'other': 2},
            ('c',): {'foo': 'bar', 'other': 3},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS,): {'foo': 'bar'},
            ('a',): {'other': 1},
            ('b',): {'other': 2},
            ('c',): {'other': 3},
        })

        # One different.
        new_metadata = {
            ('a',): {'foo': 'bar', 'other': 1},
            ('b',): {'foo': 'bar', 'other': 2},
            ('c',): {'foo': 'bar2', 'other': 3,},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            ('a',): {'foo': 'bar', 'other': 1},
            ('b',): {'foo': 'bar', 'other': 2},
            ('c',): {'foo': 'bar2', 'other': 3,},
        })

        # Recursive.
        new_metadata = {
            ('deep', 'a'): {'foo': 'bar', 'other': 1},
            ('deep', 'b'): {'foo': 'bar', 'other': 2},
            ('deep', 'c'): {'foo': 'bar', 'other': 3},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'a'): {'other': 1},
            (base.ALL_ELEMENTS, 'b'): {'other': 2},
            (base.ALL_ELEMENTS, 'c'): {'other': 3},
        })

        new_metadata = {
            ('deep', 'a'): {'foo': 'bar', 'other': 1},
            ('deep', 'b'): {'foo': 'bar', 'other': 2},
            ('deep', 'c'): {'foo': 'bar', 'other': 3},
            ('deep2', 'd'): {'foo': 'bar', 'other': 4},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'a'): {'other': 1},
            (base.ALL_ELEMENTS, 'b'): {'other': 2},
            (base.ALL_ELEMENTS, 'c'): {'other': 3},
            (base.ALL_ELEMENTS, 'd'): {'other': 4},
        })

        new_metadata = {
            ('deep', 'a'): {'foo': 'bar', 'other': 1},
            ('deep', 'b'): {'foo': 'bar', 'other': 2},
            ('deep', 'c'): {'foo': 'bar', 'other': 3},
            ('deep2', 'a'): {'foo': 'bar', 'other': 4},
            ('deep2', 'b'): {'foo': 'bar', 'other': 5},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'c'): {'other': 3},
            ('deep', 'a'): {'other': 1},
            ('deep', 'b'): {'other': 2},
            ('deep2', 'a'): {'other': 4},
            ('deep2', 'b'): {'other': 5},
        })

        new_metadata = {
            ('deep', 'a'): {'foo': 'bar', 'other': 1},
            ('deep', 'b'): {'foo': 'bar', 'other': 2},
            ('deep', 'c'): {'foo': 'bar2', 'other': 3},
            ('deep2', 'a'): {'foo': 'bar', 'other': 4},
            ('deep2', 'b'): {'foo': 'bar', 'other': 5},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 'a'): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'b'): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'c'): {'foo': 'bar2', 'other': 3},
            ('deep', 'a'): {'other': 1},
            ('deep', 'b'): {'other': 2},
            ('deep2', 'a'): {'other': 4},
            ('deep2', 'b'): {'other': 5},
        })

        new_metadata = {
            ('a', 'deep'): {'foo': 'bar', 'other': 1},
            ('b', 'deep'): {'foo': 'bar', 'other': 2},
            ('c', 'deep'): {'foo': 'bar2', 'other': 3},
            ('a', 'deep2'): {'foo': 'bar', 'other': 4},
            ('b', 'deep2'): {'foo': 'bar', 'other': 5},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            ('a', base.ALL_ELEMENTS): {'foo': 'bar'},
            ('a', 'deep'): {'other': 1},
            ('a', 'deep2'): {'other': 4},
            ('b', base.ALL_ELEMENTS): {'foo': 'bar'},
            ('b', 'deep'): {'other': 2},
            ('b', 'deep2'): {'other': 5},
            ('c', base.ALL_ELEMENTS): {'foo': 'bar2', 'other': 3},
        })

        new_metadata = {
            (base.ALL_ELEMENTS, 'a'): {'foo': 'bar', 'other': 1},
            (base.ALL_ELEMENTS, 'b'): {'foo': 'bar', 'other': 2},
            (base.ALL_ELEMENTS, 'c'): {'foo': 'bar', 'other': 3},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'a'): {'other': 1},
            (base.ALL_ELEMENTS, 'b'): {'other': 2},
            (base.ALL_ELEMENTS, 'c'): {'other': 3},
        })

        new_metadata = {
            (base.ALL_ELEMENTS, 0): {'foo': 'bar1'},
            (0, 1): {'foo': 'bar2'},
            (1, 1): {'foo': 'bar2'},
            (2, 1): {'foo': 'bar2'},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 0): {'foo': 'bar1'},
            (base.ALL_ELEMENTS, 1): {'foo': 'bar2'},
        })

        new_metadata = {
            ('deep1', 'a'): {'foo': 'bar', 'other': 1},
            ('deep1', 'b'): {'foo': 'bar2', 'other': 2},
            ('deep2', 'a'): {'foo': 'bar', 'other': 3},
            ('deep2', 'b'): {'foo': 'bar2', 'other': 4},
            ('deep3', 'a'): {'foo': 'bar', 'other': 5},
            ('deep3', 'b'): {'foo': 'bar2', 'other': 6},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 'a'): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'b'): {'foo': 'bar2'},
            ('deep1', 'a'): {'other': 1},
            ('deep1', 'b'): {'other': 2},
            ('deep2', 'a'): {'other': 3},
            ('deep2', 'b'): {'other': 4},
            ('deep3', 'a'): {'other': 5},
            ('deep3', 'b'): {'other': 6},
        })

        new_metadata = {
            ('deep1', 'a'): {'foo': 'bar', 'other': 1},
            ('deep1', 'b'): {'foo': 'bar', 'other': 2},
            ('deep2', 'c'): {'foo': 'bar', 'other': 3},
            ('deep2', 'd'): {'foo': 'bar', 'other': 4},
            ('deep3', 'e'): {'foo': 'bar', 'other': 5},
            ('deep3', 'f'): {'foo': 'bar', 'other': 6},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS): {'foo': 'bar'},
            (base.ALL_ELEMENTS, 'a'): {'other': 1},
            (base.ALL_ELEMENTS, 'b'): {'other': 2},
            (base.ALL_ELEMENTS, 'c'): {'other': 3},
            (base.ALL_ELEMENTS, 'd'): {'other': 4},
            (base.ALL_ELEMENTS, 'e'): {'other': 5},
            (base.ALL_ELEMENTS, 'f'): {'other': 6},
        })

        new_metadata = {
            ('deep1', 'a', 1): {'foo': 'bar1', 'other': 1},
            ('deep2', 'a', 2): {'foo': 'bar1', 'other': 2},
            ('deep3', 'a', 3): {'foo': 'bar1', 'other': 3},
            ('deep4', 'a', 4): {'foo': 'bar1', 'other': 4},
            ('deep1', 'b', 1): {'foo': 'bar2', 'other': 5},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 2): {'other': 2},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 3): {'other': 3},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 4): {'other': 4},
            (base.ALL_ELEMENTS, 'a', base.ALL_ELEMENTS): {'foo': 'bar1'},
            (base.ALL_ELEMENTS, 'a', 1): {'other': 1},
            (base.ALL_ELEMENTS, 'b', base.ALL_ELEMENTS): {'foo': 'bar2', 'other': 5},
        })

        new_metadata = {
            ('deep', 'a', 1): {'foo': 'bar1', 'other': 1},
            ('deep', 'a', 2): {'foo': 'bar1', 'other': 2},
            ('deep', 'b', 1): {'foo': 'bar2', 'other': 3},
            ('deep', 'b', 2): {'foo': 'bar2', 'other': 4},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 'a', base.ALL_ELEMENTS): {'foo': 'bar1'},
            (base.ALL_ELEMENTS, 'a', 1): {'other': 1},
            (base.ALL_ELEMENTS, 'a', 2): {'other': 2},
            (base.ALL_ELEMENTS, 'b', base.ALL_ELEMENTS): {'foo': 'bar2'},
            (base.ALL_ELEMENTS, 'b', 1): {'other': 3},
            (base.ALL_ELEMENTS, 'b', 2): {'other': 4},
        })

        new_metadata =  {
            ('deep', 'a', 1): {'foo': 'bar1', 'other': 'bar1'},
            ('deep', 'a', 2): {'foo': 'bar1', 'other': 'bar2'},
            ('deep', 'b', 1): {'foo': 'bar2', 'other': 'bar1'},
            ('deep', 'b', 2): {'foo': 'bar2', 'other': 'bar2'},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 1): {'other': 'bar1'},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 2): {'other': 'bar2'},
            (base.ALL_ELEMENTS, 'a', base.ALL_ELEMENTS): {'foo': 'bar1'},
            (base.ALL_ELEMENTS, 'b', base.ALL_ELEMENTS): {'foo': 'bar2'},
        })

        new_metadata = {
            ('deep1', 'a', 1): {'foo': 'bar1', 'other': 1},
            ('deep1', 'a', 2): {'foo': 'bar1', 'other': 2},
            ('deep2', 'a', 3): {'foo': 'bar1', 'other': 3},
            ('deep2', 'a', 4): {'foo': 'bar1', 'other': 4},
            ('deep1', 'b', 1): {'foo': 'bar2', 'other': 1},
            ('deep1', 'b', 2): {'foo': 'bar2', 'other': 2},
            ('deep2', 'b', 3): {'foo': 'bar2', 'other': 3},
            ('deep2', 'b', 4): {'foo': 'bar2', 'other': 4},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 1): {'other': 1},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 2): {'other': 2},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 3): {'other': 3},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 4): {'other': 4},
            (base.ALL_ELEMENTS, 'a', base.ALL_ELEMENTS): {'foo': 'bar1'},
            (base.ALL_ELEMENTS, 'b', base.ALL_ELEMENTS): {'foo': 'bar2'},
        })

        new_metadata = {
            ('deep', 'a'): {'foo': 'bar', 'other': 1},
            ('deep', 'b'): {'foo': 'bar', 'other': 2},
            ('deep2', 'b'): {'other': 3},
            ('deep2', 'c'): {'foo': 'bar', 'other': 4},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 'a'): {'other': 1},
            (base.ALL_ELEMENTS, 'c'): {'foo': 'bar', 'other': 4},
            ('deep',base.ALL_ELEMENTS): {'foo': 'bar'},
            ('deep', 'b'): {'other': 2},
            ('deep2', 'b'): {'other': 3},
        })

        new_metadata = {
            ('deep', 'a'): {'foo': 'bar', 'other': 1},
            ('deep', 'b'): {'foo': 'bar', 'other': 2},
            ('deep', 'c'): {'other': 3},
            ('deep2', 'd'): {'foo': 'bar', 'other': 4},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 'a'): {'foo': 'bar', 'other': 1},
            (base.ALL_ELEMENTS, 'b'): {'foo': 'bar', 'other': 2},
            (base.ALL_ELEMENTS, 'c'): {'other': 3},
            (base.ALL_ELEMENTS, 'd'): {'foo':'bar', 'other': 4},
        })

        new_metadata = {
            (base.ALL_ELEMENTS, 0): {'structural_type': 'numpy.int64'},
            (0, 1): {'structural_type': 'str'},
            (1, 1): {'structural_type': 'str'},
            (2, 1): {'structural_type': 'str'},
            (base.ALL_ELEMENTS, 1): {'name': 'B'},
            (0, 0): {'structural_type': 'numpy.int64'},
            (1, 0): {'structural_type': 'numpy.int64'},
            (2, 0): {'structural_type': 'numpy.int64'},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, 0): {'structural_type': 'numpy.int64'},
            (base.ALL_ELEMENTS, 1): {'name': 'B', 'structural_type': 'str'},
        })

        new_metadata = {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 0): {'structural_type': 'numpy.int64'},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 1): {'structural_type': 'str'},
            ('0', base.ALL_ELEMENTS, 0): {'name': 'A', 'structural_type': 'numpy.int64'},
            ('0', base.ALL_ELEMENTS, 1): {'name': 'B', 'structural_type': 'str'},
        }

        compacted_metadata = base.DataMetadata._compact_metadata(new_metadata, ALL_GENERATED_KEYS)

        self.assertEqual(compacted_metadata, {
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 0): {'structural_type': 'numpy.int64', 'name': 'A'},
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS, 1): {'structural_type': 'str', 'name': 'B'},
        })

    def test_greedy_prune_metadata(self):
        # Warmup test 1.
        selectors_to_compact = [('a',), ('b',),('c',)]
        compacted_selector = [(base.ALL_ELEMENTS, )]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            (base.ALL_ELEMENTS, )
        ])

        # Warmup test 2.
        selectors_to_compact = [('deep', 'a'), ('deep', 'b'), ('deep', 'c'), ('deep2', 'd')]
        compacted_selector = [(base.ALL_ELEMENTS, base.ALL_ELEMENTS,)]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            (base.ALL_ELEMENTS, base.ALL_ELEMENTS,)
        ])

        # Check if it can remove unnecessary outputs.
        selectors_to_compact = [('deep', 'a'), ('deep', 'b'), ('deep2', 'a'), ('deep2', 'b')]
        compacted_selector = [(base.ALL_ELEMENTS, 'a'), (base.ALL_ELEMENTS, 'b'), ('deep', 'a'), ('deep2', 'b')]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            (base.ALL_ELEMENTS, 'a'), (base.ALL_ELEMENTS, 'b')
        ])

        # Case when compacted_selector overlaps.
        selectors_to_compact = [('a', 'deep'), ('b', 'deep'), ('a', 'deep2'), ('b', 'deep2')]
        compacted_selector = [('a', base.ALL_ELEMENTS), ('b', base.ALL_ELEMENTS), (base.ALL_ELEMENTS, 'deep2')]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            ('a', base.ALL_ELEMENTS), ('b', base.ALL_ELEMENTS)
        ])

        # Check the order.
        selectors_to_compact = [('a', 'deep'), ('b', 'deep'), ('a', 'deep2'), ('b', 'deep2')]
        compacted_selector = [(base.ALL_ELEMENTS, 'deep2'), ('a', base.ALL_ELEMENTS), ('b', base.ALL_ELEMENTS),]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            ('a', base.ALL_ELEMENTS), ('b', base.ALL_ELEMENTS)
        ])

        # More complex compacted_selectors.
        selectors_to_compact = [('a', 'deep'), ('b', 'deep'), ('a', 'deep2'), ('b', 'deep2')]
        compacted_selector = [(base.ALL_ELEMENTS, 'deep2'), ('a', base.ALL_ELEMENTS),
                              (base.ALL_ELEMENTS, 'deep'), ('b', base.ALL_ELEMENTS),]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            (base.ALL_ELEMENTS, 'deep2'), (base.ALL_ELEMENTS, 'deep')
        ])

        # All-elements in selectors_to_compact.
        selectors_to_compact = [('deep', 'a', 1), ('deep', base.ALL_ELEMENTS, 2)]
        compacted_selector = [(base.ALL_ELEMENTS, 'a', base.ALL_ELEMENTS), ('deep', base.ALL_ELEMENTS, 2)]

        pruned_selectors = base.DataMetadata._greedy_prune_selector(compacted_selector, selectors_to_compact)

        self.assertEqual(pruned_selectors, [
            (base.ALL_ELEMENTS, 'a', base.ALL_ELEMENTS), ('deep', base.ALL_ELEMENTS, 2)
        ])

    def test_semantic_types_merge(self):
        metadata = base.DataMetadata().update(('0',), {
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'],
        })

        metadata_regular = metadata.update((base.ALL_ELEMENTS,), {
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
        })

        self.assertEqual(metadata_regular.query(('0',)).get('semantic_types', None), ('https://metadata.datadrivendiscovery.org/types/Table',))

        metadata._update_with_generated_metadata({
            (base.ALL_ELEMENTS,): {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
            },
        })

        self.assertEqual(metadata.query(('0',)).get('semantic_types', None), ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint', 'https://metadata.datadrivendiscovery.org/types/Table',))

    def test_compact(self):
        md = base.Metadata().update(('0',), {
            'key': 'value',
        })
        md = md.update(('1',), {
            'key': 'value',
        })

        md = md.compact(['key'])

        self.assertEqual(md.to_internal_json_structure(), [{
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'key': 'value'},
        }])

        md = base.Metadata().update(('0',), {
            'key': 'value',
        })
        md = md.update(('1',), {
            'key': 'value',
        })
        md = md.update(('2',), {
            'key': 'value2',
        })

        md = md.compact(['key'])

        self.assertEqual(md.to_internal_json_structure(), [{
            'selector': ['0'],
            'metadata': {'key': 'value'},
        }, {
            'selector': ['1'],
            'metadata': {'key': 'value'},
        }, {
            'selector': ['2'],
            'metadata': {'key': 'value2'},
        }])

        md = base.Metadata().update(('0',), {
            'key': 'value',
            'key2': 'value',
        })
        md = md.update(('1',), {
            'key': 'value',
            'key2': 'value',
        })

        md = md.compact(['key'])

        self.assertEqual(md.to_internal_json_structure(), [{
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {'key': 'value'},
        }, {
            'selector': ['0'],
            'metadata': {'key2': 'value'},
        }, {
            'selector': ['1'],
            'metadata': {'key2': 'value'},
        }])


if __name__ == '__main__':
    unittest.main()
