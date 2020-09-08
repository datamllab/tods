import os
import signal
import subprocess
import time
import unittest

import numpy
import pandas

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import plasma
except ModuleNotFoundError:
    plasma = None

from d3m import container


@unittest.skipIf(plasma is None, "requires Plasma")
class TestPlasma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen(['plasma_store', '-m', '1000000', '-s', '/tmp/plasma', '-d', '/dev/shm'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, encoding='utf8', preexec_fn=os.setpgrp)
        time.sleep(5)
        cls.client = plasma.connect('/tmp/plasma')

    @classmethod
    def tearDownClass(cls):
        cls.client.disconnect()
        os.killpg(os.getpgid(cls.process.pid), signal.SIGTERM)

    def test_list(self):
        l = container.List([1, 2, 3], generate_metadata=True)

        l.metadata = l.metadata.update((), {
            'test': 'foobar',
        })

        object_id = self.client.put(l)
        l_copy = self.client.get(object_id)

        self.assertIsInstance(l_copy, container.List)
        self.assertTrue(hasattr(l_copy, 'metadata'))

        self.assertSequenceEqual(l, l_copy)
        self.assertEqual(l.metadata.to_internal_json_structure(), l_copy.metadata.to_internal_json_structure())
        self.assertEqual(l_copy.metadata.query(()).get('test'), 'foobar')

    def test_ndarray(self):
        for name, dtype, values in (
                ('ints', numpy.int64, [1, 2, 3]),
                ('strings', numpy.dtype('<U1'), ['a', 'b', 'c']),
                ('objects', numpy.object, [{'a': 1}, {'b': 2}, {'c': 3}]),
        ):
            array = container.ndarray(numpy.array(values), generate_metadata=True)
            self.assertEqual(array.dtype, dtype, name)

            array.metadata = array.metadata.update((), {
                'test': 'foobar',
            })

            object_id = self.client.put(array)
            array_copy = self.client.get(object_id)

            self.assertIsInstance(array_copy, container.ndarray, name)
            self.assertTrue(hasattr(array_copy, 'metadata'), name)

            self.assertTrue(numpy.array_equal(array, array_copy), name)
            self.assertEqual(array.metadata.to_internal_json_structure(), array_copy.metadata.to_internal_json_structure(), name)
            self.assertEqual(array_copy.metadata.query(()).get('test'), 'foobar', name)

    def test_dataframe(self):
        for name, values in (
                ('ints', {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}),
                ('mix', {'A': [1, 2, 3], 'B': ['a', 'b', 'c'], 'C': [{'a':  1}, {'b': 2}, {'c': 3}]}),
        ):
            df = container.DataFrame(pandas.DataFrame(values), generate_metadata=True)

            df.metadata = df.metadata.update((), {
                'test': 'foobar',
            })

            object_id = self.client.put(df)
            df_copy = self.client.get(object_id)

            self.assertIsInstance(df_copy, container.DataFrame, name)
            self.assertTrue(hasattr(df_copy, 'metadata'), name)

            self.assertTrue(df.equals(df_copy), name)
            self.assertEqual(df.metadata.to_internal_json_structure(), df_copy.metadata.to_internal_json_structure(), name)
            self.assertEqual(df_copy.metadata.query(()).get('test'), 'foobar', name)

    def test_datasets(self):
        dataset = container.Dataset.load('sklearn://boston')

        dataset.metadata = dataset.metadata.update((), {
            'test': 'foobar',
        })

        object_id = self.client.put(dataset)
        dataset_copy = self.client.get(object_id)

        self.assertIsInstance(dataset_copy, container.Dataset)
        self.assertTrue(hasattr(dataset_copy, 'metadata'))

        self.assertEqual(len(dataset), len(dataset_copy))
        self.assertEqual(dataset.keys(), dataset_copy.keys())
        for resource_name in dataset.keys():
            self.assertTrue(numpy.array_equal(dataset[resource_name], dataset_copy[resource_name]))
        self.assertEqual(dataset.metadata.to_internal_json_structure(), dataset_copy.metadata.to_internal_json_structure())
        self.assertEqual(dataset_copy.metadata.query(()).get('test'), 'foobar')


if __name__ == '__main__':
    unittest.main()
