import builtins
import copy
import io
import json
import logging
import sys
import random
import typing
import unittest

import jsonschema
import numpy

from d3m import container, types, utils
from d3m.container import list
from d3m.metadata import base as metadata_base


class TestUtils(unittest.TestCase):
    def test_get_type_arguments(self):
        A = typing.TypeVar('A')
        B = typing.TypeVar('B')
        C = typing.TypeVar('C')

        class Base(typing.Generic[A, B]):
            pass

        class Foo(Base[A, None]):
            pass

        class Bar(Foo[A], typing.Generic[A, C]):
            pass

        class Baz(Bar[float, int]):
            pass

        self.assertEqual(utils.get_type_arguments(Bar), {
            A: typing.Any,
            B: type(None),
            C: typing.Any,
        })
        self.assertEqual(utils.get_type_arguments(Baz), {
            A: float,
            B: type(None),
            C: int,
        })

        self.assertEqual(utils.get_type_arguments(Base), {
            A: typing.Any,
            B: typing.Any,
        })

        self.assertEqual(utils.get_type_arguments(Base[float, int]), {
            A: float,
            B: int,
        })

        self.assertEqual(utils.get_type_arguments(Foo), {
            A: typing.Any,
            B: type(None),
        })

        self.assertEqual(utils.get_type_arguments(Foo[float]), {
            A: float,
            B: type(None),
        })

    def test_issubclass(self):
        self.assertTrue(utils.is_subclass(list.List, types.Container))

        T1 = typing.TypeVar('T1', bound=list.List)
        self.assertTrue(utils.is_subclass(list.List, T1))

    def test_create_enum(self):
        obj = {
            'definitions': {
                'foobar1':{
                    'type': 'array',
                    'items': {
                        'anyOf':[
                            {'enum': ['AAA']},
                            {'enum': ['BBB']},
                            {'enum': ['CCC']},
                            {'enum': ['DDD']},
                        ],
                    },
                },
                'foobar2': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'anyOf': [
                            {
                                'properties': {
                                    'type': {
                                        'type': 'string',
                                        'enum': ['EEE'],
                                    },
                                },
                            },
                            {
                                'properties': {
                                    'type': {
                                        'type': 'string',
                                        'enum': ['FFF'],
                                    },
                                },
                            },
                            {
                                'properties': {
                                    'type': {
                                        'type': 'string',
                                        'enum': ['GGG'],
                                    },
                                },
                            },
                        ],
                    },
                },
                'foobar3': {
                    'type': 'string',
                    'enum': ['HHH', 'HHH', 'III', 'JJJ'],
                }
            },
        }

        Foobar1 = utils.create_enum_from_json_schema_enum('Foobar1', obj, 'definitions.foobar1.items.anyOf[*].enum[*]')
        Foobar2 = utils.create_enum_from_json_schema_enum('Foobar2', obj, 'definitions.foobar2.items.anyOf[*].properties.type.enum[*]')
        Foobar3 = utils.create_enum_from_json_schema_enum('Foobar3', obj, 'definitions.foobar3.enum[*]')

        self.assertSequenceEqual(builtins.list(Foobar1.__members__.keys()), ['AAA', 'BBB', 'CCC', 'DDD'])
        self.assertSequenceEqual([value.value for value in Foobar1.__members__.values()], ['AAA', 'BBB', 'CCC', 'DDD'])

        self.assertSequenceEqual(builtins.list(Foobar2.__members__.keys()), ['EEE', 'FFF', 'GGG'])
        self.assertSequenceEqual([value.value for value in Foobar2.__members__.values()], ['EEE', 'FFF', 'GGG'])

        self.assertSequenceEqual(builtins.list(Foobar3.__members__.keys()), ['HHH', 'III', 'JJJ'])
        self.assertSequenceEqual([value.value for value in Foobar3.__members__.values()], ['HHH', 'III', 'JJJ'])

        self.assertTrue(Foobar1.AAA.name == 'AAA')
        self.assertTrue(Foobar1.AAA.value == 'AAA')
        self.assertTrue(Foobar1.AAA == Foobar1.AAA)
        self.assertTrue(Foobar1.AAA == 'AAA')

    def test_extendable_enum(self):
        class Foobar(utils.Enum):
            AAA = 1
            BBB = 2
            CCC = 3

        self.assertSequenceEqual(builtins.list(Foobar.__members__.keys()), ['AAA', 'BBB', 'CCC'])
        self.assertSequenceEqual([value.value for value in Foobar.__members__.values()], [1, 2, 3])

        with self.assertRaises(AttributeError):
            Foobar.register_value('CCC', 5)

        self.assertSequenceEqual(builtins.list(Foobar.__members__.keys()), ['AAA', 'BBB', 'CCC'])
        self.assertSequenceEqual([value.value for value in Foobar.__members__.values()], [1, 2, 3])

        Foobar.register_value('DDD', 4)

        self.assertSequenceEqual(builtins.list(Foobar.__members__.keys()), ['AAA', 'BBB', 'CCC', 'DDD'])
        self.assertSequenceEqual([value.value for value in Foobar.__members__.values()], [1, 2, 3, 4])

        self.assertEqual(Foobar['DDD'], 'DDD')
        self.assertEqual(Foobar(4), 'DDD')

        Foobar.register_value('EEE', 4)

        self.assertSequenceEqual(builtins.list(Foobar.__members__.keys()), ['AAA', 'BBB', 'CCC', 'DDD', 'EEE'])
        self.assertSequenceEqual([value.value for value in Foobar.__members__.values()], [1, 2, 3, 4, 4])

        self.assertEqual(Foobar['EEE'], 'DDD')
        self.assertEqual(Foobar(4), 'DDD')

    def test_redirect(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        test_stream = io.StringIO()
        sys.stdout = test_stream
        sys.stderr = test_stream

        logger = logging.getLogger('test_logger')
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            with utils.redirect_to_logging(logger=logger, pass_through=False):
                print("Test.")

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "Test.")

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            with utils.redirect_to_logging(logger=logger, pass_through=False):
                print("foo", "bar")

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "foo bar")

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            with utils.redirect_to_logging(logger=logger, pass_through=False):
                print("Test.\nTe", end="")
                print("st2.", end="")

                # The incomplete line should not be written to the logger.
                self.assertEqual(len(cm.records), 1)
                self.assertEqual(cm.records[0].message, "Test.")

        # Remaining contents should be written to logger upon closing.
        self.assertEqual(len(cm.records), 2)
        self.assertEqual(cm.records[0].message, "Test.")
        self.assertEqual(cm.records[1].message, "Test2.")

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            with utils.redirect_to_logging(logger=logger, pass_through=False):
                print("Test.  ")
                print("     ")
                print("  Test2.")
                print("    ")

        # Trailing whitespace and new lines should not be logged.
        self.assertEqual(len(cm.records), 2)
        self.assertEqual(cm.records[0].message, "Test.")
        self.assertEqual(cm.records[1].message, "  Test2.")

        logger2 = logging.getLogger('test_logger2')
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            with self.assertLogs(logger=logger2, level=logging.DEBUG) as cm2:
                with utils.redirect_to_logging(logger=logger, pass_through=True):
                    print("Test.")
                    with utils.redirect_to_logging(logger=logger2, pass_through=True):
                        print("Test2.")

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "Test.")
        self.assertEqual(len(cm2.records), 1)
        self.assertEqual(cm2.records[0].message, "Test2.")

        pass_through_lines = test_stream.getvalue().split('\n')
        self.assertEqual(len(pass_through_lines), 3)
        self.assertEqual(pass_through_lines[0], "Test.")
        self.assertEqual(pass_through_lines[1], "Test2.")
        self.assertEqual(pass_through_lines[2], "")

        records = []

        def callback(record):
            nonlocal records
            records.append(record)

        # Test recursion prevention.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            with self.assertLogs(logger=logger2, level=logging.DEBUG) as cm2:
                # We add it twice so that we test that handler does not modify record while running.
                logger2.addHandler(utils.CallbackHandler(callback))
                logger2.addHandler(utils.CallbackHandler(callback))

                with utils.redirect_to_logging(logger=logger, pass_through=False):
                    print("Test.")
                    with utils.redirect_to_logging(logger=logger2, pass_through=False):
                        # We configure handler after redirecting.
                        handler = logging.StreamHandler(sys.stdout)
                        handler.setFormatter(logging.Formatter('Test format: %(message)s'))
                        logger2.addHandler(handler)
                        print("Test2.")

        # We use outer "redirect_to_logging" to make sure nothing from inner gets out.
        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "Test.")

        self.assertEqual(len(cm2.records), 2)
        # This one comes from the print.
        self.assertEqual(cm2.records[0].message, "Test2.")
        # And this one comes from the stream handler.
        self.assertEqual(cm2.records[1].message, "Test format: Test2.")

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]['message'], "Test2.")
        self.assertEqual(records[1]['message'], "Test2.")
        self.assertEqual(records[2]['message'], "Test format: Test2.")
        self.assertEqual(records[3]['message'], "Test format: Test2.")

        test_stream.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    def test_columns_sum(self):
        dataframe = container.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, generate_metadata=True)

        dataframe_sum = utils.columns_sum(dataframe)

        self.assertEqual(dataframe_sum.values.tolist(), [[6, 15]])
        self.assertEqual(dataframe_sum.metadata.query((metadata_base.ALL_ELEMENTS, 0))['name'], 'a')
        self.assertEqual(dataframe_sum.metadata.query((metadata_base.ALL_ELEMENTS, 1))['name'], 'b')

        array = container.ndarray(dataframe, generate_metadata=True)

        array_sum = utils.columns_sum(array)

        self.assertEqual(array_sum.tolist(), [[6, 15]])
        self.assertEqual(array_sum.metadata.query((metadata_base.ALL_ELEMENTS, 0))['name'], 'a')
        self.assertEqual(array_sum.metadata.query((metadata_base.ALL_ELEMENTS, 1))['name'], 'b')

    def test_numeric(self):
        self.assertTrue(utils.is_float(type(1.0)))
        self.assertFalse(utils.is_float(type(1)))
        self.assertFalse(utils.is_int(type(1.0)))
        self.assertTrue(utils.is_int(type(1)))
        self.assertTrue(utils.is_numeric(type(1.0)))
        self.assertTrue(utils.is_numeric(type(1)))

    def test_yaml_representers(self):
        self.assertEqual(utils.yaml_load(utils.yaml_dump(numpy.int32(1))), 1)
        self.assertEqual(utils.yaml_load(utils.yaml_dump(numpy.int64(1))), 1)
        self.assertEqual(utils.yaml_load(utils.yaml_dump(numpy.float32(1.0))), 1.0)
        self.assertEqual(utils.yaml_load(utils.yaml_dump(numpy.float64(1.0))), 1.0)

    def test_json_schema_python_type(self):
        schemas = copy.copy(metadata_base.SCHEMAS)
        schemas['http://example.com/testing_python_type.json'] = {
            'id': 'http://example.com/testing_python_type.json',
            'properties': {
                'foobar': {
                    '$ref': 'https://metadata.datadrivendiscovery.org/schemas/v0/definitions.json#/definitions/python_type',
                },
            },
        }

        validator, = utils.load_schema_validators(schemas, ('testing_python_type.json',))

        validator.validate({'foobar': 'str'})
        validator.validate({'foobar': str})

        with self.assertRaisesRegex(jsonschema.exceptions.ValidationError, 'python-type'):
            validator.validate({'foobar': 1})

    def test_json_schema_numeric(self):
        schemas = copy.copy(metadata_base.SCHEMAS)
        schemas['http://example.com/testing_numeric.json'] = {
            'id': 'http://example.com/testing_numeric.json',
            'properties': {
                'int': {
                    'type': 'integer',
                },
                'float': {
                    'type': 'number',
                },
            },
        }

        validator, = utils.load_schema_validators(schemas, ('testing_numeric.json',))

        validator.validate({'float': 0})
        validator.validate({'float': 1.0})
        validator.validate({'float': 1.2})

        with self.assertRaisesRegex(jsonschema.exceptions.ValidationError, 'float'):
            validator.validate({'float': '1.2'})

        validator.validate({'int': 0})
        validator.validate({'int': 1.0})

        with self.assertRaisesRegex(jsonschema.exceptions.ValidationError, 'int'):
            validator.validate({'int': 1.2})

        with self.assertRaisesRegex(jsonschema.exceptions.ValidationError, 'int'):
            validator.validate({'int': '1.0'})

    def test_digest(self):
        self.assertEqual(utils.compute_digest({'a': 1.0, 'digest': 'xxx'}), utils.compute_digest({'a': 1.0}))
        self.assertEqual(utils.compute_hash_id({'a': 1.0, 'id': 'xxx'}), utils.compute_hash_id({'a': 1.0}))

        self.assertEqual(utils.compute_digest({'a': 1.0}), utils.compute_digest({'a': 1}))
        self.assertEqual(utils.compute_hash_id({'a': 1.0}), utils.compute_hash_id({'a': 1}))

    def test_json_equals(self):
        basic_cases = ['hello', 0, -2, 3.14, False, True, [1, 2, 3], {'a': 1}, set(['z', 'y', 'x'])]
        for case in basic_cases:
            self.assertTrue(utils.json_structure_equals(case, case))

        self.assertFalse(utils.json_structure_equals({'extra_key': 'value'}, {}))
        self.assertFalse(utils.json_structure_equals({}, {'extra_key': 'value'}))
        self.assertTrue(utils.json_structure_equals({}, {'extra_key': 'value'}, ignore_keys={'extra_key'}))

        list1 = {'a': builtins.list('type')}
        list2 = {'a': builtins.list('typo')}
        self.assertFalse(utils.json_structure_equals(list1, list2))

        json1 = {
            'a': 1,
            'b': True,
            'c': 'hello',
            'd': -2.4,
            'e': {
                'a': 'world',
            },
            'f': [
                0,
                1,
                2
            ],
            'ignore': {
                'a': False
            },
            'deep': [
                {
                    'a': {},
                    'ignore': {}
                },
                {
                    'b': [],
                    'ignore': -1
                }
            ]
        }
        json2 = {
            'a': 1,
            'b': True,
            'c': 'hello',
            'd': -2.4,
            'e': {
                'a': 'world',
            },
            'f': [
                0,
                1,
                2
            ],
            'ignore': {
                'a': True
            },
            'deep': [
                {
                    'a': {},
                    'ignore': {
                        'not_empty': 'hello world'
                    }
                },
                {
                    'b': [],
                    'ignore': 1
                }
            ]
        }

        self.assertTrue(utils.json_structure_equals(json1, json2, ignore_keys={'ignore'}))
        self.assertFalse(utils.json_structure_equals(json1, json2))

    def test_reversible_json(self):
        for obj in [
            1,
            "foobar",
            b"foobar",
            [1, 2, 3],
            [1, [2], 3],
            1.2,
            type(None),
            int,
            str,
            numpy.ndarray,
            {'foo': 'bar'},
            {'encoding': 'something', 'value': 'else'},
            metadata_base.NO_VALUE,
            metadata_base.ALL_ELEMENTS,
        ]:
            self.assertEqual(utils.from_reversible_json_structure(json.loads(json.dumps(utils.to_reversible_json_structure(obj)))), obj, str(obj))

        self.assertTrue(numpy.isnan(utils.from_reversible_json_structure(json.loads(json.dumps(utils.to_reversible_json_structure(float('nan')))))))

        self.assertEqual(utils.from_reversible_json_structure(json.loads(json.dumps(utils.to_reversible_json_structure(numpy.array([1, 2, 3]))))).tolist(), [1, 2, 3])

        with self.assertRaises(TypeError):
            utils.to_reversible_json_structure({1: 2})

    def test_global_randomness_warning(self):
        with self.assertLogs(logger=utils.logger, level=logging.DEBUG) as cm:
            with utils.global_randomness_warning():
                random.randint(0, 10)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "Using global/shared random source using 'random.randint' can make execution not reproducible.")

        with self.assertLogs(logger=utils.logger, level=logging.DEBUG) as cm:
            with utils.global_randomness_warning():
                numpy.random.randint(0, 10)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, "Using global/shared random source using 'numpy.random.randint' can make execution not reproducible.")

        if hasattr(numpy.random, 'default_rng'):
            with self.assertLogs(logger=utils.logger, level=logging.DEBUG) as cm:
                with utils.global_randomness_warning():
                    numpy.random.default_rng()

            self.assertEqual(len(cm.records), 1)
            self.assertEqual(cm.records[0].message, "Using 'numpy.random.default_rng' without a seed can make execution not reproducible.")

    def test_yaml_float_parsing(self):
        self.assertEqual(json.loads('1000.0'), 1000)
        self.assertEqual(utils.yaml_load('1000.0'), 1000)

        self.assertEqual(json.loads('1e+3'), 1000)
        self.assertEqual(utils.yaml_load('1e+3'), 1000)


if __name__ == '__main__':
    unittest.main()
