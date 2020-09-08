import json
import logging
import os
import typing
import pickle
import subprocess
import sys
import unittest
from collections import OrderedDict

import frozendict
import numpy
from sklearn.utils import validation as sklearn_validation

from d3m import container, exceptions, index, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.monomial import MonomialPrimitive
from test_primitives.random import RandomPrimitive
from test_primitives.sum import SumPrimitive
from test_primitives.increment import IncrementPrimitive


# It's defined at global scope so it can be pickled.
class TestPicklingHyperparams(hyperparams.Hyperparams):
    choice = hyperparams.Choice(
        choices={
            'alpha': hyperparams.Hyperparams.define(OrderedDict(
                value=hyperparams.Union(
                    OrderedDict(
                        float=hyperparams.Hyperparameter[float](0),
                        int=hyperparams.Hyperparameter[int](0)
                    ),
                    default='float'
                ),
            ))
        },
        default='alpha',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class TestHyperparams(unittest.TestCase):
    def test_hyperparameter(self):
        hyperparameter = hyperparams.Hyperparameter[str]('nothing')

        self.assertEqual(hyperparameter.get_default(), 'nothing')
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.sample(42), 'nothing')
        self.assertEqual(len(cm.records), 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.sample_multiple(0, 1, 42), ('nothing',))
        self.assertEqual(len(cm.records), 1)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.sample_multiple(0, 0, 42), ())
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': 'nothing',
            'semantic_types': [],
            'structural_type': str,
            'type': hyperparams.Hyperparameter,
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), 'nothing')
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.sample(42)), 'nothing')
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.get_default())), hyperparameter.get_default())
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.sample(42))), hyperparameter.sample(42))
        self.assertEqual(len(cm.records), 1)

        with self.assertRaisesRegex(TypeError, 'Value \'.*\' is not an instance of the structural type'):
            hyperparams.Hyperparameter[int]('nothing')

        with self.assertRaisesRegex(ValueError, '\'max_samples\' cannot be larger than'):
            hyperparameter.sample_multiple(0, 2, 42)

    def test_constant(self):
        hyperparameter = hyperparams.Constant(12345)

        self.assertEqual(hyperparameter.get_default(), 12345)
        self.assertEqual(hyperparameter.sample(), 12345)
        self.assertEqual(hyperparameter.sample_multiple(0, 1, 42), (12345,))

        self.assertEqual(hyperparameter.sample_multiple(0, 0, 42), ())

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': 12345,
            'semantic_types': [],
            'structural_type': int,
            'type': hyperparams.Constant,
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), 12345)
        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.sample(42)), 12345)
        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.get_default())), hyperparameter.get_default())
        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.sample(42))), hyperparameter.sample(42))

        with self.assertRaisesRegex(TypeError, 'Value \'.*\' is not an instance of the structural type'):
            hyperparams.Hyperparameter[int]('different')

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not the constant default value'):
            hyperparameter.validate(54321)

        with self.assertRaisesRegex(ValueError, '\'max_samples\' cannot be larger than'):
            self.assertEqual(hyperparameter.sample_multiple(0, 2, 42), {12345})

        hyperparameter = hyperparams.Constant('constant')

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not the constant default value'):
            hyperparameter.validate('different')

    def test_bounded(self):
        hyperparameter = hyperparams.Bounded[float](0.0, 1.0, 0.2)

        self.assertEqual(hyperparameter.get_default(), 0.2)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.sample(42), 0.37454011884736255)
        self.assertEqual(len(cm.records), 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.sample_multiple(0, 1, 7), (0.22733907982646523,))
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(hyperparameter.sample_multiple(0, 0, 42), ())

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': 0.2,
            'semantic_types': [],
            'structural_type': float,
            'type': hyperparams.Bounded,
            'lower': 0.0,
            'upper': 1.0,
            'lower_inclusive': True,
            'upper_inclusive': True,
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), 0.2)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.sample(42)), 0.37454011884736255)
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.get_default())), hyperparameter.get_default())
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.sample(42))), hyperparameter.sample(42))
        self.assertEqual(len(cm.records), 1)

        with self.assertRaisesRegex(TypeError, 'Value \'.*\' is not an instance of the structural type'):
            hyperparams.Bounded[str]('lower', 'upper', 0.2)

        with self.assertRaisesRegex(TypeError, 'Lower bound \'.*\' is not an instance of the structural type'):
            hyperparams.Bounded[str](0.0, 'upper', 'default')

        with self.assertRaisesRegex(TypeError, 'Upper bound \'.*\' is not an instance of the structural type'):
            hyperparams.Bounded[str]('lower', 1.0, 'default')

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is outside of range'):
            hyperparams.Bounded[str]('lower', 'upper', 'default')

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is outside of range'):
            hyperparams.Bounded[float](0.0, 1.0, 1.2)

        hyperparams.Bounded[typing.Optional[float]](0.0, None, 0.2)
        hyperparams.Bounded[typing.Optional[float]](None, 1.0, 0.2)

        with self.assertRaisesRegex(ValueError, 'Lower and upper bounds cannot both be None'):
            hyperparams.Bounded[typing.Optional[float]](None, None, 0.2)

        with self.assertRaisesRegex(TypeError, 'Value \'.*\' is not an instance of the structural type'):
            hyperparams.Bounded[float](0.0, 1.0, None)

        with self.assertRaises(TypeError):
            hyperparams.Bounded[typing.Optional[float]](0.0, 1.0, None)

        hyperparams.Bounded[typing.Optional[float]](None, 1.0, None)
        hyperparams.Bounded[typing.Optional[float]](0.0, None, None)

        hyperparameter = hyperparams.Bounded[float](0.0, None, 0.2)

        with self.assertRaisesRegex(ValueError, '\'max_samples\' cannot be larger than'):
            hyperparameter.sample_multiple(0, 2, 42)

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'must be finite'):
            hyperparams.Bounded[typing.Optional[float]](0.0, numpy.nan, 0)

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'must be finite'):
            hyperparams.Bounded[typing.Optional[float]](numpy.inf, 0.0, 0)

    def test_enumeration(self):
        hyperparameter = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)

        self.assertEqual(hyperparameter.get_default(), None)
        self.assertEqual(hyperparameter.sample(42), 2)
        self.assertEqual(hyperparameter.sample_multiple(0, 1, 42), ())
        self.assertEqual(hyperparameter.sample_multiple(0, 2, 42), ('b', None))
        self.assertEqual(hyperparameter.sample_multiple(0, 3, 42), ('b', None))

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': None,
            'semantic_types': [],
            'structural_type': typing.Union[str, int, type(None)],
            'type': hyperparams.Enumeration,
            'values': ['a', 'b', 1, 2, None],
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), None)
        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.sample(42)), 2)

        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.get_default())), hyperparameter.get_default())
        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.sample(42))), hyperparameter.sample(42))

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not among values'):
            hyperparams.Enumeration(['a', 'b', 1, 2], None)

        with self.assertRaisesRegex(TypeError, 'Value \'.*\' is not an instance of the structural type'):
            hyperparams.Enumeration[typing.Union[str, int]](['a', 'b', 1, 2, None], None)

        with self.assertRaisesRegex(ValueError, '\'max_samples\' cannot be larger than'):
            self.assertEqual(hyperparameter.sample_multiple(0, 6, 42), ())

        hyperparameter = hyperparams.Enumeration(['a', 'b', 'c'], 'a')

        self.assertEqual(hyperparameter.value_to_json_structure('c'), 'c')
        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure('c')), 'c')

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'contain duplicates'):
            hyperparams.Enumeration([1.0, 1], 1)

        hyperparameter = hyperparams.Enumeration([1.0, float('nan'), float('infinity'), float('-infinity')], 1.0)

        hyperparameter.validate(float('nan'))

        self.assertEqual(utils.to_json_structure(hyperparameter.to_simple_structure()), {
            'type': 'd3m.metadata.hyperparams.Enumeration',
            'default': 1.0,
            'structural_type': 'float',
            'semantic_types': [],
            'values': [1.0, 'nan', 'inf', '-inf'],
        })

        self.assertEqual(json.dumps(hyperparameter.value_to_json_structure(float('nan')), allow_nan=False), '{"encoding": "pickle", "value": "gANHf/gAAAAAAAAu"}')
        self.assertEqual(json.dumps(hyperparameter.value_to_json_structure(float('inf')), allow_nan=False), '{"encoding": "pickle", "value": "gANHf/AAAAAAAAAu"}')

    def test_other(self):
        hyperparameter = hyperparams.UniformInt(1, 10, 2)

        self.assertEqual(hyperparameter.get_default(), 2)
        self.assertEqual(hyperparameter.sample(42), 7)
        self.assertEqual(hyperparameter.sample_multiple(0, 1, 42), ())
        self.assertEqual(hyperparameter.sample_multiple(0, 2, 42), (4, 8))

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': 2,
            'semantic_types': [],
            'structural_type': int,
            'type': hyperparams.UniformInt,
            'lower': 1,
            'upper': 10,
            'lower_inclusive': True,
            'upper_inclusive': False,
        })

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is outside of range'):
            hyperparams.UniformInt(1, 10, 0)

        with self.assertRaisesRegex(ValueError, '\'max_samples\' cannot be larger than'):
            self.assertEqual(hyperparameter.sample_multiple(0, 10, 42), ())

        hyperparameter = hyperparams.Uniform(1.0, 10.0, 2.0)

        self.assertEqual(hyperparameter.get_default(), 2.0)
        self.assertEqual(hyperparameter.sample(42), 4.370861069626263)

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': 2.0,
            'semantic_types': [],
            'structural_type': float,
            'type': hyperparams.Uniform,
            'lower': 1.0,
            'upper': 10.0,
            'lower_inclusive': True,
            'upper_inclusive': False,
        })

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is outside of range'):
            hyperparams.Uniform(1.0, 10.0, 0.0)

        hyperparameter = hyperparams.LogUniform(1.0, 10.0, 2.0)

        self.assertEqual(hyperparameter.get_default(), 2.0)
        self.assertEqual(hyperparameter.sample(42), 2.368863950364078)

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': 2.0,
            'semantic_types': [],
            'structural_type': float,
            'type': hyperparams.LogUniform,
            'lower': 1.0,
            'upper': 10.0,
            'lower_inclusive': True,
            'upper_inclusive': False,
        })

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is outside of range'):
            hyperparams.LogUniform(1.0, 10.0, 0.0)

        hyperparameter = hyperparams.UniformBool(True)

        self.assertEqual(hyperparameter.get_default(), True)
        self.assertEqual(hyperparameter.sample(42), True)

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': True,
            'semantic_types': [],
            'structural_type': bool,
            'type': hyperparams.UniformBool,
        })

        with self.assertRaises(exceptions.InvalidArgumentValueError):
            hyperparams.UniformInt(0, 1, 1, lower_inclusive=False, upper_inclusive=False)

        hyperparameter = hyperparams.UniformInt(0, 2, 1, lower_inclusive=False, upper_inclusive=False)

        self.assertEqual(hyperparameter.sample(42), 1)

        with self.assertRaises(exceptions.InvalidArgumentValueError):
            hyperparameter.sample_multiple(2, 2, 42)

        self.assertEqual(hyperparameter.sample_multiple(2, 2, 42, with_replacement=True), (1, 1))

    def test_union(self):
        hyperparameter = hyperparams.Union(
            OrderedDict(
                none=hyperparams.Hyperparameter(None),
                range=hyperparams.UniformInt(1, 10, 2)
            ),
            'none',
        )

        self.assertEqual(hyperparameter.get_default(), None)
        self.assertEqual(hyperparameter.sample(45), 4)

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': None,
            'semantic_types': [],
            'structural_type': typing.Optional[int],
            'type': hyperparams.Union,
            'configuration': {
                'none': {
                    'default': None,
                    'semantic_types': [],
                    'structural_type': type(None),
                    'type': hyperparams.Hyperparameter,
                },
                'range': {
                    'default': 2,
                    'semantic_types': [],
                    'structural_type': int,
                    'type': hyperparams.UniformInt,
                    'lower': 1,
                    'upper': 10,
                    'lower_inclusive': True,
                    'upper_inclusive': False,
                }
            }
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), {'case': 'none', 'value': None})
        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.sample(45)), {'case': 'range', 'value': 4})

        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.get_default())), hyperparameter.get_default())
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.sample(42))), hyperparameter.sample(42))
        self.assertEqual(len(cm.records), 1)

        with self.assertRaisesRegex(TypeError, 'Hyper-parameter name is not a string'):
            hyperparams.Union(OrderedDict({1: hyperparams.Hyperparameter(None)}), 1)

        with self.assertRaisesRegex(TypeError, 'Hyper-parameter description is not an instance of the Hyperparameter class'):
            hyperparams.Union(OrderedDict(none=None), 'none')

        with self.assertRaisesRegex(ValueError, 'Default value \'.*\' is not in configuration'):
            hyperparams.Union(OrderedDict(range=hyperparams.UniformInt(1, 10, 2)), 'none')

        hyperparams.Union(OrderedDict(range=hyperparams.UniformInt(1, 10, 2), default=hyperparams.Hyperparameter('nothing')), 'default')
        hyperparams.Union[typing.Union[str, int]](OrderedDict(range=hyperparams.UniformInt(1, 10, 2), default=hyperparams.Hyperparameter('nothing')), 'default')

        with self.assertRaisesRegex(TypeError, 'Hyper-parameter \'.*\' is not a subclass of the structural type'):
            hyperparams.Union[str](OrderedDict(range=hyperparams.UniformInt(1, 10, 2), default=hyperparams.Hyperparameter('nothing')), 'default')

    def test_hyperparams(self):
        class TestHyperparams(hyperparams.Hyperparams):
            a = hyperparams.Union(OrderedDict(
                range=hyperparams.UniformInt(1, 10, 2),
                none=hyperparams.Hyperparameter(None),
            ), 'range')
            b = hyperparams.Uniform(1.0, 10.0, 2.0)

        testCls = hyperparams.Hyperparams.define(OrderedDict(
            a=hyperparams.Union(OrderedDict(
                range=hyperparams.UniformInt(1, 10, 2),
                none=hyperparams.Hyperparameter(None),
            ), 'range'),
            b=hyperparams.Uniform(1.0, 10.0, 2.0),
        ), set_names=True)

        for cls in (TestHyperparams, testCls):
            self.assertEqual(cls.configuration['a'].name, 'a', cls)

            self.assertEqual(cls.defaults(), {'a': 2, 'b': 2.0}, cls)
            self.assertEqual(cls.defaults(), cls({'a': 2, 'b': 2.0}), cls)
            self.assertEqual(cls.sample(42), {'a': 4, 'b': 9.556428757689245}, cls)
            self.assertEqual(cls.sample(42), cls({'a': 4, 'b': 9.556428757689245}), cls)
            self.assertEqual(cls(cls.defaults(), b=3.0), {'a': 2, 'b': 3.0}, cls)
            self.assertEqual(cls(cls.defaults(), **{'b': 4.0}), {'a': 2, 'b': 4.0}, cls)
            self.assertEqual(cls.defaults('a'), 2, cls)
            self.assertEqual(cls.defaults('b'), 2.0, cls)

            self.assertEqual(cls.to_simple_structure(), {
                'a': {
                    'default': 2,
                    'semantic_types': [],
                    'structural_type': typing.Optional[int],
                    'type': hyperparams.Union,
                    'configuration': {
                        'none': {
                            'default': None,
                            'semantic_types': [],
                            'structural_type': type(None),
                            'type': hyperparams.Hyperparameter,
                        },
                        'range': {
                            'default': 2,
                            'lower': 1,
                            'semantic_types': [],
                            'structural_type': int,
                            'type': hyperparams.UniformInt,
                            'upper': 10,
                            'lower_inclusive': True,
                            'upper_inclusive': False,
                        },
                    },
                },
                'b': {
                    'default': 2.0,
                    'semantic_types': [],
                    'structural_type': float,
                    'type': hyperparams.Uniform,
                    'lower': 1.0,
                    'upper': 10.0,
                    'lower_inclusive': True,
                    'upper_inclusive': False,
                }
            }, cls)

            test_hyperparams = cls({'a': cls.configuration['a'].get_default(), 'b': cls.configuration['b'].get_default()})

            self.assertEqual(test_hyperparams['a'], 2, cls)
            self.assertEqual(test_hyperparams['b'], 2.0, cls)

            self.assertEqual(test_hyperparams.values_to_json_structure(), {'a': {'case': 'range', 'value': 2}, 'b': 2.0})
            self.assertEqual(cls.values_from_json_structure(test_hyperparams.values_to_json_structure()), test_hyperparams)

            with self.assertRaisesRegex(ValueError, 'Not all hyper-parameters are specified', msg=cls):
                cls({'a': cls.configuration['a'].get_default()})

            with self.assertRaisesRegex(ValueError, 'Additional hyper-parameters are specified', msg=cls):
                cls({'a': cls.configuration['a'].get_default(), 'b': cls.configuration['b'].get_default(), 'c': 'two'})

            cls({'a': 3, 'b': 3.0})
            cls({'a': None, 'b': 3.0})

            test_hyperparams = cls(a=None, b=3.0)
            self.assertEqual(test_hyperparams['a'], None, cls)
            self.assertEqual(test_hyperparams['b'], 3.0, cls)

            with self.assertRaisesRegex(ValueError, 'Value \'.*\' for hyper-parameter \'.*\' has not validated with any of configured hyper-parameters', msg=cls):
                cls({'a': 0, 'b': 3.0})

            with self.assertRaisesRegex(ValueError, 'Value \'.*\' for hyper-parameter \'.*\' is outside of range', msg=cls):
                cls({'a': 3, 'b': 100.0})

            class SubTestHyperparams(cls):
                c = hyperparams.Hyperparameter[int](0)

            self.assertEqual(SubTestHyperparams.defaults(), {'a': 2, 'b': 2.0, 'c': 0}, cls)

            testSubCls = cls.define(OrderedDict(
                c=hyperparams.Hyperparameter[int](0),
            ), set_names=True)

            self.assertEqual(testSubCls.defaults(), {'a': 2, 'b': 2.0, 'c': 0}, cls)

        class ConfigurationHyperparams(hyperparams.Hyperparams):
            configuration = hyperparams.Uniform(1.0, 10.0, 2.0)

        self.assertEqual(ConfigurationHyperparams.configuration['configuration'].to_simple_structure(), hyperparams.Uniform(1.0, 10.0, 2.0).to_simple_structure())

    def test_numpy(self):
        class TestHyperparams(hyperparams.Hyperparams):
            value = hyperparams.Hyperparameter[container.ndarray](
                default=container.ndarray([0], generate_metadata=True),
            )

        values = TestHyperparams(value=container.ndarray([1, 2, 3], generate_metadata=True))

        self.assertEqual(values.values_to_json_structure(), {'value': {'encoding': 'pickle', 'value': 'gANjbnVtcHkuY29yZS5tdWx0aWFycmF5Cl9yZWNvbnN0cnVjdApxAGNkM20uY29udGFpbmVyLm51bXB5Cm5kYXJyYXkKcQFLAIVxAkMBYnEDh3EEUnEFfXEGKFgFAAAAbnVtcHlxByhLAUsDhXEIY251bXB5CmR0eXBlCnEJWAIAAABpOHEKSwBLAYdxC1JxDChLA1gBAAAAPHENTk5OSv////9K/////0sAdHEOYolDGAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAHEPdHEQWAgAAABtZXRhZGF0YXERY2QzbS5tZXRhZGF0YS5iYXNlCkRhdGFNZXRhZGF0YQpxEimBcRN9cRQoWBEAAABfY3VycmVudF9tZXRhZGF0YXEVY2QzbS5tZXRhZGF0YS5iYXNlCk1ldGFkYXRhRW50cnkKcRYpgXEXTn1xGChYCAAAAGVsZW1lbnRzcRljZDNtLnV0aWxzCnBtYXAKcRp9cRuFcRxScR1YDAAAAGFsbF9lbGVtZW50c3EeaBYpgXEfTn1xIChoGWgdaB5OaBFjZnJvemVuZGljdApGcm96ZW5PcmRlcmVkRGljdApxISmBcSJ9cSMoWAUAAABfZGljdHEkY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnElKVJxJlgPAAAAc3RydWN0dXJhbF90eXBlcSdjbnVtcHkKaW50NjQKcShzWAUAAABfaGFzaHEpTnViWAgAAABpc19lbXB0eXEqiVgRAAAAaXNfZWxlbWVudHNfZW1wdHlxK4h1hnEsYmgRaCEpgXEtfXEuKGgkaCUpUnEvKFgGAAAAc2NoZW1hcTBYQgAAAGh0dHBzOi8vbWV0YWRhdGEuZGF0YWRyaXZlbmRpc2NvdmVyeS5vcmcvc2NoZW1hcy92MC9jb250YWluZXIuanNvbnExaCdoAVgJAAAAZGltZW5zaW9ucTJoISmBcTN9cTQoaCRoJSlScTVYBgAAAGxlbmd0aHE2SwNzaClOdWJ1aClOdWJoKoloK4h1hnE3YmgpTnVidWIu'}})
        self.assertTrue(numpy.array_equal(TestHyperparams.values_from_json_structure(values.values_to_json_structure())['value'], values['value']))

    def test_set(self):
        set_hyperparameter = hyperparams.Set(hyperparams.Hyperparameter[int](1), [])
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(set(set_hyperparameter.sample_multiple(min_samples=2, max_samples=2)), {(1,), ()})
        self.assertEqual(len(cm.records), 1)
        elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
        set_hyperparameter = hyperparams.Set(elements, ('a', 'b', 1, 2, None), 5, 5)

        self.assertEqual(set_hyperparameter.get_default(), ('a', 'b', 1, 2, None))
        self.assertEqual(set_hyperparameter.sample(45), ('b', None, 'a', 1, 2))
        self.assertEqual(set_hyperparameter.get_max_samples(), 1)
        self.assertEqual(set_hyperparameter.sample_multiple(1, 1, 42), (('b', None, 1, 'a', 2),))
        self.assertEqual(set_hyperparameter.sample_multiple(0, 1, 42), ())

        self.maxDiff = None

        self.assertEqual(set_hyperparameter.to_simple_structure(), {
            'default': ('a', 'b', 1, 2, None),
            'semantic_types': [],
            'structural_type': typing.Sequence[typing.Union[str, int, type(None)]],
            'type': hyperparams.Set,
            'min_size': 5,
            'max_size': 5,
            'elements': {
                'default': None,
                'semantic_types': [],
                'structural_type': typing.Union[str, int, type(None)],
                'type': hyperparams.Enumeration,
                'values': ['a', 'b', 1, 2, None],
            },
            'is_configuration': False,
        })

        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default()), ['a', 'b', 1, 2, None])
        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(45)), ['b', None, 'a', 1, 2])

        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default())), set_hyperparameter.get_default())
        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(45))), set_hyperparameter.sample(45))

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' has less than 5 elements'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.Set(elements, (), 5, 5)

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not among values'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.Set(elements, ('a', 'b', 1, 2, 3), 5, 5)

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' has duplicate elements'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.Set(elements, ('a', 'b', 1, 2, 2), 5, 5)

        set_hyperparameter.contribute_to_class('foo')

        with self.assertRaises(KeyError):
            set_hyperparameter.get_default('foo')

        list_of_supported_metafeatures = ['f1', 'f2', 'f3']
        metafeature = hyperparams.Enumeration(list_of_supported_metafeatures, list_of_supported_metafeatures[0], semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])
        set_hyperparameter = hyperparams.Set(metafeature, (), 0, 3)

        self.assertEqual(set_hyperparameter.get_default(), ())
        self.assertEqual(set_hyperparameter.sample(42), ('f2', 'f3'))
        self.assertEqual(set_hyperparameter.get_max_samples(), 8)
        self.assertEqual(set_hyperparameter.sample_multiple(0, 3, 42), (('f2', 'f3', 'f1'), ('f2', 'f3')))

        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default()), [])
        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(42)), ['f2', 'f3'])

        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default())), set_hyperparameter.get_default())
        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(42))), set_hyperparameter.sample(42))

        set_hyperparameter = hyperparams.Set(metafeature, (), 0, None)

        self.assertEqual(set_hyperparameter.get_default(), ())
        self.assertEqual(set_hyperparameter.sample(42), ('f2', 'f3'))
        self.assertEqual(set_hyperparameter.get_max_samples(), 8)
        self.assertEqual(set_hyperparameter.sample_multiple(0, 3, 42), (('f2', 'f3', 'f1'), ('f2', 'f3')))

    def test_set_with_hyperparams(self):
        elements = hyperparams.Hyperparams.define(OrderedDict(
            range=hyperparams.UniformInt(1, 10, 2),
            enum=hyperparams.Enumeration(['a', 'b', 1, 2, None], None),
        ))
        set_hyperparameter = hyperparams.Set(elements, (elements(range=2, enum='a'),), 0, 5)

        self.assertEqual(set_hyperparameter.get_default(), ({'range': 2, 'enum': 'a'},))
        self.assertEqual(set_hyperparameter.sample(45), ({'range': 4, 'enum': None}, {'range': 1, 'enum': 2}, {'range': 5, 'enum': 'b'}))
        self.assertEqual(set_hyperparameter.get_max_samples(), 1385980)
        self.assertEqual(set_hyperparameter.sample_multiple(1, 1, 42), (({'range': 8, 'enum': None}, {'range': 5, 'enum': 'b'}, {'range': 3, 'enum': 1}),))
        self.assertEqual(set_hyperparameter.sample_multiple(0, 1, 42), ())
        self.maxDiff = None

        self.assertEqual(set_hyperparameter.to_simple_structure(), {
            'default': ({'range': 2, 'enum': 'a'},),
            'elements': {
                'enum': {
                    'default': None,
                    'semantic_types': [],
                    'structural_type': typing.Union[str, int, type(None)],
                    'type': hyperparams.Enumeration,
                    'values': ['a', 'b', 1, 2, None],
                },
                'range': {
                    'default': 2,
                    'lower': 1,
                    'semantic_types': [],
                    'structural_type': int,
                    'type': hyperparams.UniformInt,
                    'upper': 10,
                    'lower_inclusive': True,
                    'upper_inclusive': False,
                },
            },
            'is_configuration': True,
            'max_size': 5,
            'min_size': 0,
            'semantic_types': [],
            'structural_type': typing.Sequence[elements],
            'type': hyperparams.Set,
        })

        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default()), [{'range': 2, 'enum': 'a'}])
        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(45)), [{'range': 4, 'enum': None}, {'range': 1, 'enum': 2}, {'range': 5, 'enum': 'b'}])

        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default())), set_hyperparameter.get_default())
        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(45))), set_hyperparameter.sample(45))

        # We have to explicitly disable setting names if we want to use it for "Set" hyper-parameter.
        class SetHyperparams(hyperparams.Hyperparams, set_names=False):
            choice = hyperparams.Choice({
                'none': hyperparams.Hyperparams,
                'range': hyperparams.Hyperparams.define(OrderedDict(
                    value=hyperparams.UniformInt(1, 10, 2),
                )),
            }, 'none')

        class TestHyperparams(hyperparams.Hyperparams):
            a = set_hyperparameter
            b = hyperparams.Set(SetHyperparams, (SetHyperparams({'choice': {'choice': 'none'}}),), 0, 3)

        self.assertEqual(TestHyperparams.to_simple_structure(), {
            'a': {
                'type': hyperparams.Set,
                'default': ({'range': 2, 'enum': 'a'},),
                'structural_type': typing.Sequence[elements],
                'semantic_types': [],
                'elements': {
                    'range': {
                        'type': hyperparams.UniformInt,
                        'default': 2,
                        'structural_type': int,
                        'semantic_types': [],
                        'lower': 1,
                        'upper': 10,
                        'lower_inclusive': True,
                        'upper_inclusive': False,
                    },
                    'enum': {
                        'type': hyperparams.Enumeration,
                        'default': None,
                        'structural_type': typing.Union[str, int, type(None)],
                        'semantic_types': [],
                        'values': ['a', 'b', 1, 2, None],
                    },
                },
                'is_configuration': True,
                'min_size': 0,
                'max_size': 5,
            },
            'b': {
                'type': hyperparams.Set,
                'default': ({'choice': {'choice': 'none'}},),
                'structural_type': typing.Sequence[SetHyperparams],
                'semantic_types': [],
                'elements': {
                    'choice': {
                        'type': hyperparams.Choice,
                        'default': {'choice': 'none'},
                        'structural_type': typing.Dict,
                        'semantic_types': [],
                        'choices': {
                            'none': {
                                'choice': {
                                    'type': hyperparams.Hyperparameter,
                                    'default': 'none',
                                    'structural_type': str,
                                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
                                },
                            },
                            'range': {
                                'value': {
                                    'type': hyperparams.UniformInt,
                                    'default': 2,
                                    'structural_type': int,
                                    'semantic_types': [],
                                    'lower': 1,
                                    'upper': 10,
                                    'lower_inclusive': True,
                                    'upper_inclusive': False,
                                },
                                'choice': {
                                    'type': hyperparams.Hyperparameter,
                                    'default': 'range',
                                    'structural_type': str,
                                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
                                },
                            },
                        },
                    },
                },
                'is_configuration': True,
                'min_size': 0,
                'max_size': 3,
            },
        })

        self.assertEqual(TestHyperparams.configuration['b'].elements.configuration['choice'].choices['range'].configuration['value'].name, 'b.choice.range.value')

        self.assertEqual(TestHyperparams.defaults(), {
            'a': ({'range': 2, 'enum': 'a'},),
            'b': ({'choice': {'choice': 'none'}},),
        })
        self.assertTrue(utils.is_instance(TestHyperparams.defaults()['a'], typing.Sequence[elements]))
        self.assertTrue(utils.is_instance(TestHyperparams.defaults()['b'], typing.Sequence[SetHyperparams]))

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.sample(42), {
                'a': ({'range': 8, 'enum': None}, {'range': 5, 'enum': 'b'}, {'range': 3, 'enum': 1}),
                'b': (
                    {
                        'choice': {'value': 5, 'choice': 'range'},
                    }, {
                        'choice': {'value': 8, 'choice': 'range'},
                    },
                ),
            })
        self.assertEqual(len(cm.records), 1)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.sample(42).values_to_json_structure(), {
                'a': [{'range': 8, 'enum': None}, {'range': 5, 'enum': 'b'}, {'range': 3, 'enum': 1}],
                'b': [
                    {
                        'choice': {'value': 5, 'choice': 'range'},
                    }, {
                        'choice': {'value': 8, 'choice': 'range'},
                    },
                ],
            })
        self.assertEqual(len(cm.records), 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.values_from_json_structure(TestHyperparams.sample(42).values_to_json_structure()), TestHyperparams.sample(42))
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(len(list(TestHyperparams.traverse())), 8)

        self.assertEqual(TestHyperparams.defaults('a'), ({'range': 2, 'enum': 'a'},))
        self.assertEqual(TestHyperparams.defaults('a.range'), 2)
        # Default of a whole "Set" hyper-parameter can be different than of nested hyper-parameters.
        self.assertEqual(TestHyperparams.defaults('a.enum'), None)
        self.assertEqual(TestHyperparams.defaults('b'), ({'choice': {'choice': 'none'}},))
        self.assertEqual(TestHyperparams.defaults('b.choice'), {'choice': 'none'})
        self.assertEqual(TestHyperparams.defaults('b.choice.none'), {'choice': 'none'})
        self.assertEqual(TestHyperparams.defaults('b.choice.none.choice'), 'none')
        self.assertEqual(TestHyperparams.defaults('b.choice.range'), {'choice': 'range', 'value': 2})
        self.assertEqual(TestHyperparams.defaults('b.choice.range.value'), 2)
        self.assertEqual(TestHyperparams.defaults('b.choice.range.choice'), 'range')

        self.assertEqual(TestHyperparams(TestHyperparams.defaults(), b=(
            SetHyperparams({
                'choice': {'value': 5, 'choice': 'range'},
            }),
            SetHyperparams({
                'choice': {'value': 8, 'choice': 'range'},
            }),
        )), {
            'a': ({'range': 2, 'enum': 'a'},),
            'b': (
                {
                    'choice': {'value': 5, 'choice': 'range'},
                },
                {
                    'choice': {'value': 8, 'choice': 'range'},
                },
            ),
        })
        self.assertEqual(TestHyperparams(TestHyperparams.defaults(), **{'a': (
            elements({'range': 8, 'enum': None}),
            elements({'range': 5, 'enum': 'b'}),
            elements({'range': 3, 'enum': 1}),
        )}), {
            'a': (
                {'range': 8, 'enum': None},
                {'range': 5, 'enum': 'b'},
                {'range': 3, 'enum': 1},
            ),
            'b': ({'choice': {'choice': 'none'}},)
        })

        self.assertEqual(TestHyperparams.defaults().replace({'a': (
            elements({'range': 8, 'enum': None}),
            elements({'range': 5, 'enum': 'b'}),
            elements({'range': 3, 'enum': 1}),
        )}), {
            'a': (
                {'range': 8, 'enum': None},
                {'range': 5, 'enum': 'b'},
                {'range': 3, 'enum': 1},
            ),
            'b': ({'choice': {'choice': 'none'}},),
        })

    def test_choice(self):
        choices_hyperparameter = hyperparams.Choice({
            'none': hyperparams.Hyperparams,
            'range': hyperparams.Hyperparams.define(OrderedDict(
                # To test that we can use this name.
                configuration=hyperparams.UniformInt(1, 10, 2),
            )),
        }, 'none')

        # Class should not be changed directly (when adding "choice").
        self.assertEqual(hyperparams.Hyperparams.configuration, {})

        self.assertEqual(choices_hyperparameter.get_default(), {'choice': 'none'})
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(choices_hyperparameter.sample(45), {'choice': 'range', 'configuration': 4})
        self.assertEqual(len(cm.records), 1)
        self.assertEqual(choices_hyperparameter.get_max_samples(), 10)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(choices_hyperparameter.sample_multiple(0, 3, 42), (frozendict.frozendict({'choice': 'range', 'configuration': 8}), frozendict.frozendict({'choice': 'none'})))
        self.assertEqual(len(cm.records), 1)

        self.maxDiff = None

        self.assertEqual(choices_hyperparameter.to_simple_structure(), {
            'default': {'choice': 'none'},
            'semantic_types': [],
            'structural_type': typing.Dict,
            'type': hyperparams.Choice,
            'choices': {
                'none': {
                    'choice': {
                        'default': 'none',
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
                        'structural_type': str,
                        'type': hyperparams.Hyperparameter,
                    },
                },
                'range': {
                    'choice': {
                        'default': 'range',
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
                        'structural_type': str,
                        'type': hyperparams.Hyperparameter,
                    },
                    'configuration': {
                        'default': 2,
                        'lower': 1,
                        'lower_inclusive': True,
                        'upper': 10,
                        'upper_inclusive': False,
                        'semantic_types': [],
                        'structural_type': int,
                        'type': hyperparams.UniformInt,
                    },
                },
            },
        })

        self.assertEqual(choices_hyperparameter.value_to_json_structure(choices_hyperparameter.get_default()), {'choice': 'none'})
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(choices_hyperparameter.value_to_json_structure(choices_hyperparameter.sample(45)), {'configuration': 4, 'choice': 'range'})
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(choices_hyperparameter.value_from_json_structure(choices_hyperparameter.value_to_json_structure(choices_hyperparameter.get_default())), choices_hyperparameter.get_default())
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(choices_hyperparameter.value_from_json_structure(choices_hyperparameter.value_to_json_structure(choices_hyperparameter.sample(45))), choices_hyperparameter.sample(45))
        self.assertEqual(len(cm.records), 1)

        # We have to explicitly disable setting names if we want to use it for "Choice" hyper-parameter.
        class ChoicesHyperparams(hyperparams.Hyperparams, set_names=False):
            foo = hyperparams.UniformInt(5, 20, 10)

        class TestHyperparams(hyperparams.Hyperparams):
            a = choices_hyperparameter
            b = hyperparams.Choice({
                'nochoice': ChoicesHyperparams,
            }, 'nochoice')

        self.assertEqual(TestHyperparams.configuration['a'].choices['range'].configuration['configuration'].name, 'a.range.configuration')

        self.assertEqual(TestHyperparams.defaults(), {'a': {'choice': 'none'}, 'b': {'choice': 'nochoice', 'foo': 10}})
        self.assertIsInstance(TestHyperparams.defaults()['a'], hyperparams.Hyperparams)
        self.assertIsInstance(TestHyperparams.defaults()['b'], ChoicesHyperparams)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.sample(42), {'a': {'choice': 'none'}, 'b': {'choice': 'nochoice', 'foo': 8}})
        self.assertEqual(len(cm.records), 1)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.sample(42).values_to_json_structure(), {'a': {'choice': 'none'}, 'b': {'choice': 'nochoice', 'foo': 8}})
        self.assertEqual(len(cm.records), 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.values_from_json_structure(TestHyperparams.sample(42).values_to_json_structure()), TestHyperparams.sample(42))
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(len(list(TestHyperparams.traverse())), 7)

        self.assertEqual(TestHyperparams.defaults('a'), {'choice': 'none'})
        self.assertEqual(TestHyperparams.defaults('a.none'), {'choice': 'none'})
        self.assertEqual(TestHyperparams.defaults('a.none.choice'), 'none')
        self.assertEqual(TestHyperparams.defaults('a.range'), {'choice': 'range', 'configuration': 2})
        self.assertEqual(TestHyperparams.defaults('a.range.configuration'), 2)
        self.assertEqual(TestHyperparams.defaults('a.range.choice'), 'range')
        self.assertEqual(TestHyperparams.defaults('b'), {'choice': 'nochoice', 'foo': 10})
        self.assertEqual(TestHyperparams.defaults('b.nochoice'), {'choice': 'nochoice', 'foo': 10})
        self.assertEqual(TestHyperparams.defaults('b.nochoice.foo'), 10)
        self.assertEqual(TestHyperparams.defaults('b.nochoice.choice'), 'nochoice')

    def test_primitive(self):
        # To hide any logging or stdout output.
        with utils.silence():
            index.register_primitive('d3m.primitives.regression.monomial.Test', MonomialPrimitive)
            index.register_primitive('d3m.primitives.data_generation.random.Test', RandomPrimitive)
            index.register_primitive('d3m.primitives.operator.sum.Test', SumPrimitive)
            index.register_primitive('d3m.primitives.operator.increment.Test', IncrementPrimitive)

        hyperparameter = hyperparams.Primitive(MonomialPrimitive)

        self.assertEqual(hyperparameter.structural_type, MonomialPrimitive)
        self.assertEqual(hyperparameter.get_default(), MonomialPrimitive)
        # To hide any logging or stdout output.
        with utils.silence():
            self.assertEqual(hyperparameter.sample(42), MonomialPrimitive)

        hyperparams_class = MonomialPrimitive.metadata.get_hyperparams()
        primitive = MonomialPrimitive(hyperparams=hyperparams_class.defaults())

        hyperparameter = hyperparams.Enumeration([MonomialPrimitive, RandomPrimitive, SumPrimitive, IncrementPrimitive, None], None)

        self.assertEqual(hyperparameter.structural_type, typing.Union[MonomialPrimitive, RandomPrimitive, SumPrimitive, IncrementPrimitive, type(None)])
        self.assertEqual(hyperparameter.get_default(), None)
        self.assertEqual(hyperparameter.sample(42), IncrementPrimitive)

        hyperparameter = hyperparams.Enumeration[typing.Optional[base.PrimitiveBase]]([MonomialPrimitive, RandomPrimitive, SumPrimitive, IncrementPrimitive, None], None)

        self.assertEqual(hyperparameter.structural_type, typing.Optional[base.PrimitiveBase])
        self.assertEqual(hyperparameter.get_default(), None)
        self.assertEqual(hyperparameter.sample(42), IncrementPrimitive)

        set_hyperparameter = hyperparams.Set(hyperparameter, (MonomialPrimitive, RandomPrimitive), 2, 4)

        self.assertEqual(set_hyperparameter.get_default(), (MonomialPrimitive, RandomPrimitive))
        self.assertEqual(set_hyperparameter.sample(42), (RandomPrimitive, None, SumPrimitive, MonomialPrimitive))

        union_hyperparameter = hyperparams.Union(OrderedDict(
            none=hyperparams.Hyperparameter(None),
            primitive=hyperparams.Enumeration[base.PrimitiveBase]([MonomialPrimitive, RandomPrimitive, SumPrimitive, IncrementPrimitive], MonomialPrimitive),
        ), 'none')

        self.assertEqual(union_hyperparameter.get_default(), None)
        self.assertEqual(union_hyperparameter.sample(45), SumPrimitive)

        hyperparameter = hyperparams.Enumeration([primitive, RandomPrimitive, SumPrimitive, IncrementPrimitive, None], None)

        self.assertEqual(hyperparameter.structural_type, typing.Union[MonomialPrimitive, RandomPrimitive, SumPrimitive, IncrementPrimitive, type(None)])
        self.assertEqual(hyperparameter.get_default(), None)
        self.assertEqual(hyperparameter.sample(42), IncrementPrimitive)

        hyperparameter = hyperparams.Enumeration[typing.Optional[base.PrimitiveBase]]([primitive, RandomPrimitive, SumPrimitive, IncrementPrimitive, None], None)

        self.assertEqual(hyperparameter.structural_type, typing.Optional[base.PrimitiveBase])
        self.assertEqual(hyperparameter.get_default(), None)
        self.assertEqual(hyperparameter.sample(42), IncrementPrimitive)

        set_hyperparameter = hyperparams.Set(hyperparameter, (primitive, RandomPrimitive), 2, 4)

        self.assertEqual(set_hyperparameter.get_default(), (primitive, RandomPrimitive))
        self.assertEqual(set_hyperparameter.sample(42), (RandomPrimitive, None, SumPrimitive, primitive))

        union_hyperparameter = hyperparams.Union(OrderedDict(
            none=hyperparams.Hyperparameter(None),
            primitive=hyperparams.Enumeration[base.PrimitiveBase]([primitive, RandomPrimitive, SumPrimitive, IncrementPrimitive], primitive),
        ), 'none')

        self.assertEqual(union_hyperparameter.get_default(), None)
        self.assertEqual(union_hyperparameter.sample(45), SumPrimitive)

        hyperparameter = hyperparams.Primitive(primitive)

        self.assertEqual(hyperparameter.structural_type, MonomialPrimitive)
        self.assertEqual(hyperparameter.get_default(), primitive)
        # To hide any logging or stdout output.
        with utils.silence():
            self.assertEqual(hyperparameter.sample(42), primitive)

        hyperparameter = hyperparams.Primitive[base.PrimitiveBase](MonomialPrimitive)

        self.assertEqual(hyperparameter.get_default(), MonomialPrimitive)
        # To hide any logging or stdout output.
        with utils.silence():
            # There might be additional primitives available in the system,
            # so we cannot know which one will really be returned.
            self.assertTrue(hyperparameter.sample(42), hyperparameter.matching_primitives)

        self.maxDiff = None

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': MonomialPrimitive,
            'semantic_types': [],
            'structural_type': base.PrimitiveBase,
            'type': hyperparams.Primitive,
            'primitive_families': [],
            'algorithm_types': [],
            'produce_methods': [],
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), {'class': 'd3m.primitives.regression.monomial.Test'})
        self.assertEqual(hyperparameter.value_from_json_structure(hyperparameter.value_to_json_structure(hyperparameter.get_default())), hyperparameter.get_default())

        self.assertTrue(hyperparameter.get_max_samples() >= 4, hyperparameter.get_max_samples())

        hyperparameter = hyperparams.Primitive[base.PrimitiveBase](primitive)

        self.assertEqual(hyperparameter.get_default(), primitive)

        self.assertEqual(hyperparameter.to_simple_structure(), {
            'default': primitive,
            'semantic_types': [],
            'structural_type': base.PrimitiveBase,
            'type': hyperparams.Primitive,
            'primitive_families': [],
            'algorithm_types': [],
            'produce_methods': [],
        })

        self.assertEqual(hyperparameter.value_to_json_structure(hyperparameter.get_default()), {'instance': 'gANjdGVzdF9wcmltaXRpdmVzLm1vbm9taWFsCk1vbm9taWFsUHJpbWl0aXZlCnEAKYFxAX1xAihYCwAAAGNvbnN0cnVjdG9ycQN9cQQoWAsAAABoeXBlcnBhcmFtc3EFY3Rlc3RfcHJpbWl0aXZlcy5tb25vbWlhbApIeXBlcnBhcmFtcwpxBimBcQd9cQhYBAAAAGJpYXNxCUcAAAAAAAAAAHNiWAsAAAByYW5kb21fc2VlZHEKSwB1WAYAAABwYXJhbXNxC2N0ZXN0X3ByaW1pdGl2ZXMubW9ub21pYWwKUGFyYW1zCnEMKYFxDVgBAAAAYXEOSwBzdWIu'})

        set_hyperparameter = hyperparams.Set(hyperparameter, (MonomialPrimitive, RandomPrimitive), 2, 4)

        self.assertEqual(set_hyperparameter.get_default(), (MonomialPrimitive, RandomPrimitive))

        union_hyperparameter = hyperparams.Union(OrderedDict(
            none=hyperparams.Hyperparameter(None),
            primitive=hyperparameter,
        ), 'none')

        self.assertEqual(union_hyperparameter.get_default(), None)

    def test_invalid_name(self):
        with self.assertRaisesRegex(ValueError, 'Hyper-parameter name \'.*\' contains invalid characters.'):
            hyperparams.Hyperparams.define({
                'foo.bar': hyperparams.Uniform(1.0, 10.0, 2.0),
            })

    def test_class_as_default(self):
        class Foo:
            pass

        foo = Foo()

        hyperparameter = hyperparams.Enumeration(['a', 'b', 1, 2, foo], foo)

        self.assertEqual(hyperparameter.value_to_json_structure(1), {'encoding': 'pickle', 'value': 'gANLAS4='})

        hyperparameter = hyperparams.Enumeration(['a', 'b', 1, 2], 2)

        self.assertEqual(hyperparameter.value_to_json_structure(1), 1)

    def test_configuration_immutability(self):
        class TestHyperparams(hyperparams.Hyperparams):
            a = hyperparams.Union(OrderedDict(
                range=hyperparams.UniformInt(1, 10, 2),
                none=hyperparams.Hyperparameter(None),
            ), 'range')
            b = hyperparams.Uniform(1.0, 10.0, 2.0)

        with self.assertRaisesRegex(TypeError, '\'FrozenOrderedDict\' object does not support item assignment'):
            TestHyperparams.configuration['c'] = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)

        with self.assertRaisesRegex(AttributeError, 'Hyper-parameters configuration is immutable'):
            TestHyperparams.configuration = OrderedDict(
                range=hyperparams.UniformInt(1, 10, 2),
                none=hyperparams.Hyperparameter(None),
            )

    def test_dict_as_default(self):
        Inputs = container.DataFrame
        Outputs = container.DataFrame

        class Hyperparams(hyperparams.Hyperparams):
            value = hyperparams.Hyperparameter({}, semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])

        # Silence any validation warnings.
        with utils.silence():
            class Primitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': '152ea984-d8a4-4a37-87a0-29829b082e54',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'python_path': 'd3m.primitives.test.dict_as_default',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

        self.assertEqual(Primitive.metadata.query()['primitive_code']['hyperparams']['value']['default'], {})

    def test_comma_warning(self):
        logger = logging.getLogger('d3m.metadata.hyperparams')

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            class Hyperparams(hyperparams.Hyperparams):
                value = hyperparams.Hyperparameter({}, semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']),

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].message, 'Probably invalid definition of a hyper-parameter. Hyper-parameter should be defined as class attribute without a trailing comma.')

    def test_json_schema(self):
        Inputs = container.DataFrame
        Outputs = container.DataFrame

        # Silence any validation warnings.
        with utils.silence():
            # Defining primitive triggers checking against JSON schema.
            class TestJsonPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, TestPicklingHyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': 'cdfada09-5161-4f2e-bc7f-223d843d59c1',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'python_path': 'd3m.primitives.test.json_schema',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

    def test_pickling(self):
        pickle.loads(pickle.dumps(TestPicklingHyperparams))

        unpickled = pickle.loads(pickle.dumps(TestPicklingHyperparams.defaults()))

        self.assertEqual(unpickled['choice'].configuration['value'].structural_type, typing.Union[float, int])

    def test_sorted_set(self):
        set_hyperparameter = hyperparams.SortedSet(hyperparams.Hyperparameter[int](1), [])
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(set(set_hyperparameter.sample_multiple(min_samples=2, max_samples=2)), {(1,), ()})
        self.assertEqual(len(cm.records), 1)

        elements = hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e')
        set_hyperparameter = hyperparams.SortedSet(elements, ('a', 'b', 'c', 'd', 'e'), 5, 5)

        self.assertEqual(set_hyperparameter.get_default(), ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(set_hyperparameter.sample(45), ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(set_hyperparameter.get_max_samples(), 1)
        self.assertEqual(set_hyperparameter.sample_multiple(1, 1, 42), (('a', 'b', 'c', 'd', 'e'),))
        self.assertEqual(set_hyperparameter.sample_multiple(0, 1, 42), ())

        self.maxDiff = None

        self.assertEqual(set_hyperparameter.to_simple_structure(), {
            'default': ('a', 'b', 'c', 'd', 'e'),
            'semantic_types': [],
            'structural_type': typing.Sequence[str],
            'type': hyperparams.SortedSet,
            'min_size': 5,
            'max_size': 5,
            'elements': {
                'default': 'e',
                'semantic_types': [],
                'structural_type': str,
                'type': hyperparams.Enumeration,
                'values': ['a', 'b', 'c', 'd', 'e'],
            },
            'ascending': True,
        })

        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default()), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(45)), ['a', 'b', 'c', 'd', 'e'])

        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default())), set_hyperparameter.get_default())
        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(45))), set_hyperparameter.sample(45))

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' has less than 5 elements'):
            elements = hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e')
            hyperparams.SortedSet(elements, (), 5, 5)

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not among values'):
            elements = hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e')
            hyperparams.SortedSet(elements, ('a', 'b', 'c', 'd', 'f'), 5, 5)

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' has duplicate elements'):
            elements = hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e')
            hyperparams.SortedSet(elements, ('a', 'b', 'c', 'd', 'd'), 5, 5)

        set_hyperparameter.contribute_to_class('foo')

        with self.assertRaises(KeyError):
            set_hyperparameter.get_default('foo')

        list_of_supported_metafeatures = ['f1', 'f2', 'f3']
        metafeature = hyperparams.Enumeration(list_of_supported_metafeatures, list_of_supported_metafeatures[0], semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])
        set_hyperparameter = hyperparams.SortedSet(metafeature, (), 0, 3)

        self.assertEqual(set_hyperparameter.get_default(), ())
        self.assertEqual(set_hyperparameter.sample(42), ('f2', 'f3'))
        self.assertEqual(set_hyperparameter.get_max_samples(), 8)
        self.assertEqual(set_hyperparameter.sample_multiple(0, 3, 42), (('f1', 'f2', 'f3'), ('f2', 'f3')))

        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default()), [])
        self.assertEqual(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(42)), ['f2', 'f3'])

        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.get_default())), set_hyperparameter.get_default())
        self.assertEqual(set_hyperparameter.value_from_json_structure(set_hyperparameter.value_to_json_structure(set_hyperparameter.sample(42))), set_hyperparameter.sample(42))

        set_hyperparameter = hyperparams.SortedSet(metafeature, (), 0, None)

        self.assertEqual(set_hyperparameter.get_default(), ())
        self.assertEqual(set_hyperparameter.sample(42), ('f2', 'f3'))
        self.assertEqual(set_hyperparameter.get_max_samples(), 8)
        self.assertEqual(set_hyperparameter.sample_multiple(0, 3, 42), (('f1', 'f2', 'f3'), ('f2', 'f3')))

        set_hyperparameter = hyperparams.SortedSet(hyperparams.Hyperparameter[int](0), (0, 1), min_size=2, max_size=2)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(set_hyperparameter.sample_multiple(1, 1, 42), ((0, 1),))
        self.assertEqual(len(cm.records), 1)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(set_hyperparameter.sample(42), (0, 1))
        self.assertEqual(len(cm.records), 1)

        set_hyperparameter = hyperparams.SortedSet(hyperparams.Hyperparameter[int](0), (0,), min_size=1, max_size=1)

        with self.assertLogs(hyperparams.logger) as cm:
            set_hyperparameter.sample(42)
        self.assertEqual(len(cm.records), 1)

        set_hyperparameter = hyperparams.SortedSet(hyperparams.Uniform(0.0, 100.0, 50.0, lower_inclusive=False, upper_inclusive=False), (25.0, 75.0), min_size=2, max_size=2)

        self.assertEqual(set_hyperparameter.sample(42), (37.454011884736246, 95.07143064099162))

    def test_sorted_set_with_hyperparams(self):
        elements = hyperparams.Hyperparams.define(OrderedDict(
            range=hyperparams.UniformInt(1, 10, 2),
            enum=hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e'),
        ))

        with self.assertRaises(exceptions.NotSupportedError):
            hyperparams.SortedSet(elements, (elements(range=2, enum='a'),), 0, 5)

    def test_list(self):
        list_hyperparameter = hyperparams.List(hyperparams.Hyperparameter[int](1), [], 0, 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(set(list_hyperparameter.sample_multiple(min_samples=2, max_samples=2)), {(1,), ()})
        self.assertEqual(len(cm.records), 1)

        elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
        list_hyperparameter = hyperparams.List(elements, ('a', 'b', 1, 2, None), 5, 5)

        self.assertEqual(list_hyperparameter.get_default(), ('a', 'b', 1, 2, None))
        self.assertEqual(list_hyperparameter.sample(45), (2, 2, None, 'a', 2))
        self.assertEqual(list_hyperparameter.get_max_samples(), 3125)
        self.assertEqual(list_hyperparameter.sample_multiple(1, 1, 42), ((2, None, 1, None, None),))
        self.assertEqual(list_hyperparameter.sample_multiple(0, 1, 42), ())

        self.maxDiff = None

        self.assertEqual(list_hyperparameter.to_simple_structure(), {
            'default': ('a', 'b', 1, 2, None),
            'semantic_types': [],
            'structural_type': typing.Sequence[typing.Union[str, int, type(None)]],
            'type': hyperparams.List,
            'min_size': 5,
            'max_size': 5,
            'elements': {
                'default': None,
                'semantic_types': [],
                'structural_type': typing.Union[str, int, type(None)],
                'type': hyperparams.Enumeration,
                'values': ['a', 'b', 1, 2, None],
            },
            'is_configuration': False,
        })

        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default()), ['a', 'b', 1, 2, None])
        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(45)), [2, 2, None, 'a', 2])

        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default())), list_hyperparameter.get_default())
        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(45))), list_hyperparameter.sample(45))

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' has less than 5 elements'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.List(elements, (), 5, 5)

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not among values'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.List(elements, ('a', 'b', 1, 2, 3), 5, 5)

        list_hyperparameter.contribute_to_class('foo')

        with self.assertRaises(KeyError):
            list_hyperparameter.get_default('foo')

        list_of_supported_metafeatures = ['f1', 'f2', 'f3']
        metafeature = hyperparams.Enumeration(list_of_supported_metafeatures, list_of_supported_metafeatures[0], semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])
        list_hyperparameter = hyperparams.List(metafeature, (), 0, 3)

        self.assertEqual(list_hyperparameter.get_default(), ())
        self.assertEqual(list_hyperparameter.sample(42), ('f1', 'f3'))
        self.assertEqual(list_hyperparameter.get_max_samples(), 40)
        self.assertEqual(list_hyperparameter.sample_multiple(0, 3, 42), (('f1', 'f3', 'f3'), ('f1', 'f1', 'f3')))

        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default()), [])
        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(42)), ['f1', 'f3'])

        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default())), list_hyperparameter.get_default())
        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(42))), list_hyperparameter.sample(42))

        list_hyperparameter = hyperparams.List(metafeature, (), 0, 10)

        self.assertEqual(list_hyperparameter.get_default(), ())
        self.assertEqual(list_hyperparameter.sample(42), ('f1', 'f3', 'f3', 'f1', 'f1', 'f3'))
        self.assertEqual(list_hyperparameter.get_max_samples(), 88573)
        self.assertEqual(list_hyperparameter.sample_multiple(0, 3, 42), (('f1', 'f3', 'f3'), ('f1', 'f1', 'f3', 'f2', 'f3', 'f3', 'f3')))

        list_hyperparameter = hyperparams.List(hyperparams.Bounded(1, None, 100), (100,), min_size=1, max_size=None)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(list_hyperparameter.sample(42), (100,))
        self.assertEqual(len(cm.records), 1)

    def test_list_with_hyperparams(self):
        elements = hyperparams.Hyperparams.define(OrderedDict(
            range=hyperparams.UniformInt(1, 10, 2),
            enum=hyperparams.Enumeration(['a', 'b', 1, 2, None], None),
        ))
        list_hyperparameter = hyperparams.List(elements, (elements(range=2, enum='a'),), 0, 5)

        self.assertEqual(list_hyperparameter.get_default(), ({'range': 2, 'enum': 'a'},))
        self.assertEqual(list_hyperparameter.sample(45), ({'range': 4, 'enum': None}, {'range': 1, 'enum': 2}, {'range': 5, 'enum': 'b'}))
        self.assertEqual(list_hyperparameter.get_max_samples(), 188721946)
        self.assertEqual(list_hyperparameter.sample_multiple(1, 1, 42), (({'range': 8, 'enum': None}, {'range': 5, 'enum': 'b'}, {'range': 3, 'enum': 1}),))
        self.assertEqual(list_hyperparameter.sample_multiple(0, 1, 42), ())
        self.maxDiff = None

        self.assertEqual(list_hyperparameter.to_simple_structure(), {
            'default': ({'range': 2, 'enum': 'a'},),
            'elements': {
                'enum': {
                    'default': None,
                    'semantic_types': [],
                    'structural_type': typing.Union[str, int, type(None)],
                    'type': hyperparams.Enumeration,
                    'values': ['a', 'b', 1, 2, None],
                },
                'range': {
                    'default': 2,
                    'lower': 1,
                    'semantic_types': [],
                    'structural_type': int,
                    'type': hyperparams.UniformInt,
                    'upper': 10,
                    'lower_inclusive': True,
                    'upper_inclusive': False,
                },
            },
            'is_configuration': True,
            'max_size': 5,
            'min_size': 0,
            'semantic_types': [],
            'structural_type': typing.Sequence[elements],
            'type': hyperparams.List,
        })

        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default()), [{'range': 2, 'enum': 'a'}])
        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(45)), [{'range': 4, 'enum': None}, {'range': 1, 'enum': 2}, {'range': 5, 'enum': 'b'}])

        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default())), list_hyperparameter.get_default())
        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(45))), list_hyperparameter.sample(45))

        # We have to explicitly disable setting names if we want to use it for "List" hyper-parameter.
        class ListHyperparams(hyperparams.Hyperparams, set_names=False):
            choice = hyperparams.Choice({
                'none': hyperparams.Hyperparams,
                'range': hyperparams.Hyperparams.define(OrderedDict(
                    value=hyperparams.UniformInt(1, 10, 2),
                )),
            }, 'none')

        class TestHyperparams(hyperparams.Hyperparams):
            a = list_hyperparameter
            b = hyperparams.List(ListHyperparams, (ListHyperparams({'choice': {'choice': 'none'}}),), 0, 3)

        self.assertEqual(TestHyperparams.to_simple_structure(), {
            'a': {
                'type': hyperparams.List,
                'default': ({'range': 2, 'enum': 'a'},),
                'structural_type': typing.Sequence[elements],
                'semantic_types': [],
                'elements': {
                    'range': {
                        'type': hyperparams.UniformInt,
                        'default': 2,
                        'structural_type': int,
                        'semantic_types': [],
                        'lower': 1,
                        'upper': 10,
                        'lower_inclusive': True,
                        'upper_inclusive': False,
                    },
                    'enum': {
                        'type': hyperparams.Enumeration,
                        'default': None,
                        'structural_type': typing.Union[str, int, type(None)],
                        'semantic_types': [],
                        'values': ['a', 'b', 1, 2, None],
                    },
                },
                'is_configuration': True,
                'min_size': 0,
                'max_size': 5,
            },
            'b': {
                'type': hyperparams.List,
                'default': ({'choice': {'choice': 'none'}},),
                'structural_type': typing.Sequence[ListHyperparams],
                'semantic_types': [],
                'elements': {
                    'choice': {
                        'type': hyperparams.Choice,
                        'default': {'choice': 'none'},
                        'structural_type': typing.Dict,
                        'semantic_types': [],
                        'choices': {
                            'none': {
                                'choice': {
                                    'type': hyperparams.Hyperparameter,
                                    'default': 'none',
                                    'structural_type': str,
                                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
                                },
                            },
                            'range': {
                                'value': {
                                    'type': hyperparams.UniformInt,
                                    'default': 2,
                                    'structural_type': int,
                                    'semantic_types': [],
                                    'lower': 1,
                                    'upper': 10,
                                    'lower_inclusive': True,
                                    'upper_inclusive': False,
                                },
                                'choice': {
                                    'type': hyperparams.Hyperparameter,
                                    'default': 'range',
                                    'structural_type': str,
                                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/ChoiceParameter'],
                                },
                            },
                        },
                    },
                },
                'is_configuration': True,
                'min_size': 0,
                'max_size': 3,
            },
        })

        self.assertEqual(TestHyperparams.configuration['b'].elements.configuration['choice'].choices['range'].configuration['value'].name, 'b.choice.range.value')

        self.assertEqual(TestHyperparams.defaults(), {
            'a': ({'range': 2, 'enum': 'a'},),
            'b': ({'choice': {'choice': 'none'}},),
        })
        self.assertTrue(utils.is_instance(TestHyperparams.defaults()['a'], typing.Sequence[elements]))
        self.assertTrue(utils.is_instance(TestHyperparams.defaults()['b'], typing.Sequence[ListHyperparams]))

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.sample(42), {
                'a': ({'range': 8, 'enum': None}, {'range': 5, 'enum': 'b'}, {'range': 3, 'enum': 1}),
                'b': (
                    {
                        'choice': {'value': 5, 'choice': 'range'},
                    }, {
                        'choice': {'value': 8, 'choice': 'range'},
                    },
                ),
            })
        self.assertEqual(len(cm.records), 1)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.sample(42).values_to_json_structure(), {
                'a': [{'range': 8, 'enum': None}, {'range': 5, 'enum': 'b'}, {'range': 3, 'enum': 1}],
                'b': [
                    {
                        'choice': {'value': 5, 'choice': 'range'},
                    }, {
                        'choice': {'value': 8, 'choice': 'range'},
                    },
                ],
            })
        self.assertEqual(len(cm.records), 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(TestHyperparams.values_from_json_structure(TestHyperparams.sample(42).values_to_json_structure()), TestHyperparams.sample(42))
        self.assertEqual(len(cm.records), 1)

        self.assertEqual(len(list(TestHyperparams.traverse())), 8)

        self.assertEqual(TestHyperparams.defaults('a'), ({'range': 2, 'enum': 'a'},))
        self.assertEqual(TestHyperparams.defaults('a.range'), 2)
        # Default of a whole "List" hyper-parameter can be different than of nested hyper-parameters.
        self.assertEqual(TestHyperparams.defaults('a.enum'), None)
        self.assertEqual(TestHyperparams.defaults('b'), ({'choice': {'choice': 'none'}},))
        self.assertEqual(TestHyperparams.defaults('b.choice'), {'choice': 'none'})
        self.assertEqual(TestHyperparams.defaults('b.choice.none'), {'choice': 'none'})
        self.assertEqual(TestHyperparams.defaults('b.choice.none.choice'), 'none')
        self.assertEqual(TestHyperparams.defaults('b.choice.range'), {'choice': 'range', 'value': 2})
        self.assertEqual(TestHyperparams.defaults('b.choice.range.value'), 2)
        self.assertEqual(TestHyperparams.defaults('b.choice.range.choice'), 'range')

        self.assertEqual(TestHyperparams(TestHyperparams.defaults(), b=(
            ListHyperparams({
                'choice': {'value': 5, 'choice': 'range'},
            }),
            ListHyperparams({
                'choice': {'value': 8, 'choice': 'range'},
            }),
        )), {
            'a': ({'range': 2, 'enum': 'a'},),
            'b': (
                {
                    'choice': {'value': 5, 'choice': 'range'},
                },
                {
                    'choice': {'value': 8, 'choice': 'range'},
                },
            ),
        })
        self.assertEqual(TestHyperparams(TestHyperparams.defaults(), **{'a': (
            elements({'range': 8, 'enum': None}),
            elements({'range': 5, 'enum': 'b'}),
            elements({'range': 3, 'enum': 1}),
        )}), {
            'a': (
                {'range': 8, 'enum': None},
                {'range': 5, 'enum': 'b'},
                {'range': 3, 'enum': 1},
            ),
            'b': ({'choice': {'choice': 'none'}},)
        })

        self.assertEqual(TestHyperparams.defaults().replace({'a': (
            elements({'range': 8, 'enum': None}),
            elements({'range': 5, 'enum': 'b'}),
            elements({'range': 3, 'enum': 1}),
        )}), {
            'a': (
                {'range': 8, 'enum': None},
                {'range': 5, 'enum': 'b'},
                {'range': 3, 'enum': 1},
            ),
            'b': ({'choice': {'choice': 'none'}},),
        })

    def test_sorted_list(self):
        list_hyperparameter = hyperparams.SortedList(hyperparams.Hyperparameter[int](1), [], 0, 1)
        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(set(list_hyperparameter.sample_multiple(min_samples=2, max_samples=2)), {(1,), ()})
        self.assertEqual(len(cm.records), 1)

        elements = hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e')
        list_hyperparameter = hyperparams.SortedList(elements, ('a', 'b', 'c', 'd', 'e'), 5, 5)

        self.assertEqual(list_hyperparameter.get_default(), ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(list_hyperparameter.sample(45), ('a', 'd', 'd', 'd', 'e'))
        self.assertEqual(list_hyperparameter.get_max_samples(), 126)
        self.assertEqual(list_hyperparameter.sample_multiple(1, 1, 42), (('c', 'd', 'e', 'e', 'e'),))
        self.assertEqual(list_hyperparameter.sample_multiple(0, 1, 42), ())

        self.maxDiff = None

        self.assertEqual(list_hyperparameter.to_simple_structure(), {
            'default': ('a', 'b', 'c', 'd', 'e'),
            'semantic_types': [],
            'structural_type': typing.Sequence[str],
            'type': hyperparams.SortedList,
            'min_size': 5,
            'max_size': 5,
            'elements': {
                'default': 'e',
                'semantic_types': [],
                'structural_type': str,
                'type': hyperparams.Enumeration,
                'values': ['a', 'b', 'c', 'd', 'e'],
            },
            'ascending': True,
        })

        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default()), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(45)), ['a', 'd', 'd', 'd', 'e'])

        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default())), list_hyperparameter.get_default())
        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(45))), list_hyperparameter.sample(45))

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' has less than 5 elements'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.SortedList(elements, (), 5, 5)

        with self.assertRaisesRegex(ValueError, 'Value \'.*\' is not among values'):
            elements = hyperparams.Enumeration(['a', 'b', 1, 2, None], None)
            hyperparams.SortedList(elements, ('a', 'b', 1, 2, 3), 5, 5)

        list_hyperparameter.contribute_to_class('foo')

        with self.assertRaises(KeyError):
            list_hyperparameter.get_default('foo')

        list_of_supported_metafeatures = ['f1', 'f2', 'f3']
        metafeature = hyperparams.Enumeration(list_of_supported_metafeatures, list_of_supported_metafeatures[0], semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'])
        list_hyperparameter = hyperparams.SortedList(metafeature, (), 0, 3)

        self.assertEqual(list_hyperparameter.get_default(), ())
        self.assertEqual(list_hyperparameter.sample(42), ('f1', 'f3'))
        self.assertEqual(list_hyperparameter.get_max_samples(), 20)
        self.assertEqual(list_hyperparameter.sample_multiple(0, 3, 42), (('f1', 'f3', 'f3'), ('f1', 'f1', 'f3')))

        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default()), [])
        self.assertEqual(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(42)), ['f1', 'f3'])

        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.get_default())), list_hyperparameter.get_default())
        self.assertEqual(list_hyperparameter.value_from_json_structure(list_hyperparameter.value_to_json_structure(list_hyperparameter.sample(42))), list_hyperparameter.sample(42))

        list_hyperparameter = hyperparams.SortedList(metafeature, (), 0, 10)

        self.assertEqual(list_hyperparameter.get_default(), ())
        self.assertEqual(list_hyperparameter.sample(42), ('f1', 'f1', 'f1', 'f3', 'f3', 'f3'))
        self.assertEqual(list_hyperparameter.get_max_samples(), 286)
        self.assertEqual(list_hyperparameter.sample_multiple(0, 3, 42), (('f1', 'f3', 'f3'), ('f1', 'f1', 'f2', 'f3', 'f3', 'f3', 'f3')))

        list_hyperparameter = hyperparams.SortedList(hyperparams.Bounded[int](1, None, 1), (1, 1), min_size=2, max_size=2)

        with self.assertLogs(hyperparams.logger) as cm:
            self.assertEqual(list_hyperparameter.sample(42), (1, 1))
        self.assertEqual(len(cm.records), 1)

    def test_sorted_list_with_hyperparams(self):
        elements = hyperparams.Hyperparams.define(OrderedDict(
            range=hyperparams.UniformInt(1, 10, 2),
            enum=hyperparams.Enumeration(['a', 'b', 'c', 'd', 'e'], 'e'),
        ))

        with self.assertRaises(exceptions.NotSupportedError):
            hyperparams.SortedList(elements, (elements(range=2, enum='a'),), 0, 5)

    def test_import_cycle(self):
        # All references to "hyperparams_module" in "d3m.metadata.base" should be lazy:
        # for example, as a string in the typing signature, because we have an import cycle.
        subprocess.run([sys.executable, '-c', 'import d3m.metadata.base'], check=True)
        subprocess.run([sys.executable, '-c', 'import d3m.metadata.hyperparams'], check=True)

    def test_union_float_int(self):
        float_hp = hyperparams.Uniform(1, 10, 2)
        int_hp = hyperparams.UniformInt(1, 10, 2)

        x = float_hp.value_from_json_structure(2.0)
        self.assertEqual(x, 2.0)
        self.assertIs(type(x), float)

        x = float_hp.value_from_json_structure(2)
        self.assertEqual(x, 2.0)
        self.assertIs(type(x), float)

        x = float_hp.value_from_json_structure(2.1)
        self.assertEqual(x, 2.1)
        self.assertIs(type(x), float)

        x = int_hp.value_from_json_structure(2.0)
        self.assertEqual(x, 2)
        self.assertIs(type(x), int)

        x = int_hp.value_from_json_structure(2)
        self.assertEqual(x, 2)
        self.assertIs(type(x), int)

        with self.assertRaises(exceptions.InvalidArgumentTypeError):
            int_hp.value_from_json_structure(2.1)

        hyperparameter = hyperparams.Union(
            OrderedDict(
                float=hyperparams.Uniform(1, 5, 2),
                int=hyperparams.UniformInt(6, 10, 7),
            ),
            'float',
        )

        self.assertEqual(hyperparameter.value_to_json_structure(2.0), {'case': 'float', 'value': 2.0})
        self.assertEqual(hyperparameter.value_to_json_structure(7), {'case': 'int', 'value': 7})

        x = hyperparameter.value_from_json_structure({'case': 'float', 'value': 2.0})
        self.assertEqual(x, 2.0)
        self.assertIs(type(x), float)

        x = hyperparameter.value_from_json_structure({'case': 'float', 'value': 2.1})
        self.assertEqual(x, 2.1)
        self.assertIs(type(x), float)

        x = hyperparameter.value_from_json_structure({'case': 'float', 'value': 2})
        self.assertEqual(x, 2.0)
        self.assertIs(type(x), float)

        x = hyperparameter.value_from_json_structure({'case': 'int', 'value': 7})
        self.assertEqual(x, 7)
        self.assertIs(type(x), int)

        x = hyperparameter.value_from_json_structure({'case': 'int', 'value': 7.0})
        self.assertEqual(x, 7)
        self.assertIs(type(x), int)

    def test_can_serialize_to_json(self):
        # See: https://gitlab.com/datadrivendiscovery/d3m/-/issues/440
        # This is enumeration internally so it tests also that enumeration values are kept as-is when sampled.
        hyperparameter = hyperparams.UniformBool(True)
        sample = hyperparameter.sample()
        self.assertIsInstance(sample, bool)
        x = hyperparameter.value_to_json_structure(sample)
        json.dumps(x)

    def test_sampling_type(self):
        sample = hyperparams.Uniform(0, 10, 5).sample()
        self.assertIsInstance(sample, float)

    def test_numpy_sampling(self):
        class UniformInt64(hyperparams.Bounded[numpy.int64]):
            def __init__(
                self, lower: numpy.int64, upper: numpy.int64, default: numpy.int64, *, lower_inclusive: bool = True, upper_inclusive: bool = False,
                semantic_types: typing.Sequence[str] = None, description: str = None,
            ) -> None:
                if lower is None or upper is None:
                    raise exceptions.InvalidArgumentValueError("Bounds cannot be None.")

                super().__init__(lower, upper, default, lower_inclusive=lower_inclusive, upper_inclusive=upper_inclusive, semantic_types=semantic_types, description=description)

            def _initialize_effective_bounds(self) -> None:
                self._initialize_effective_bounds_int()

                super()._initialize_effective_bounds()

            def sample(self, random_state: numpy.random.RandomState = None) -> int:
                random_state = sklearn_validation.check_random_state(random_state)

                return self.structural_type(random_state.randint(self._effective_lower, self._effective_upper))

            def get_max_samples(self) -> typing.Optional[int]:
                return self._effective_upper - self._effective_lower

        with self.assertRaises(exceptions.InvalidArgumentTypeError):
            UniformInt64(0, 10, 5)

        hyperparameter = UniformInt64(numpy.int64(0), numpy.int64(10), numpy.int64(5))
        sample = hyperparameter.sample()
        self.assertIsInstance(sample, numpy.int64)
        x = hyperparameter.value_to_json_structure(sample)
        json.dumps(x)


if __name__ == '__main__':
    unittest.main()
