import typing
import unittest

import numpy

from d3m.metadata import params
from d3m import container


class TestParams(unittest.TestCase):
    def test_params(self):
        class TestParams(params.Params):
            a: str
            b: int

        test_params = TestParams({'a': 'foo', 'b': 42})

        self.assertEqual(test_params['a'], 'foo')
        self.assertEqual(test_params['b'], 42)

        with self.assertRaisesRegex(ValueError, 'Not all parameters are specified'):
            TestParams({'a': 'foo'})

        with self.assertRaisesRegex(ValueError, 'Additional parameters are specified'):
            TestParams({'a': 'foo', 'b': 42, 'c': None})

        test_params = TestParams(a='bar', b=10)
        self.assertEqual(test_params['a'], 'bar')
        self.assertEqual(test_params['b'], 10)

        with self.assertRaisesRegex(TypeError, 'Value \'.*\' is not an instance of the type'):
            TestParams({'a': 'foo', 'b': 10.1})

        with self.assertRaisesRegex(TypeError, 'Only methods and attribute type annotations can be defined on Params class'):
            class ErrorParams(params.Params):
                a = str
                b = int

    def test_numpy(self):
        class TestParams(params.Params):
            state: container.ndarray

        TestParams(state=container.ndarray([1, 2, 3], generate_metadata=True))

    def test_list_int64(self):
        class TestParams(params.Params):
            mapping: typing.Dict

        TestParams(mapping={'a': [numpy.int64(1), numpy.int64(1)]})


if __name__ == '__main__':
    unittest.main()
