import importlib
import inspect
import json
import pkgutil
import typing
import unittest

import numpy

from d3m import container, primitive_interfaces, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer


class TestPrimitiveMetadata(unittest.TestCase):
    # Test more complicated hyper parameters.
    def test_hyperparms(self):
        Inputs = container.List
        Outputs = container.List

        class Hyperparams(hyperparams.Hyperparams):
            n_components = hyperparams.Hyperparameter[typing.Optional[int]](
                default=None,
                description='Number of components (< n_classes - 1) for dimensionality reduction.',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
            learning_rate = hyperparams.Uniform(
                lower=0.01,
                upper=2,
                default=0.1,
                description='Learning rate shrinks the contribution of each classifier by ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.',
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/TuningParameter',
                    'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
                ],
            )
            array1 = hyperparams.Hyperparameter[container.ndarray](
                default=container.ndarray(numpy.array([[1, 2], [3, 4]]), generate_metadata=True),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
            array2 = hyperparams.Hyperparameter[container.DataFrame](
                default=container.DataFrame([[1, 2], [3, 4]], generate_metadata=True),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )

        # Silence any validation warnings.
        with utils.silence():
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

        self.assertEqual(TestPrimitive.metadata.query()['primitive_code'].get('hyperparams', {}), {
            'n_components': {
                'type': hyperparams.Hyperparameter,
                'default': None,
                'structural_type': typing.Optional[int],
                'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TuningParameter',),
                'description': 'Number of components (< n_classes - 1) for dimensionality reduction.',
            },
            'learning_rate': {
                'type': hyperparams.Uniform,
                'default': 0.1,
                'structural_type': float,
                'semantic_types': (
                    'https://metadata.datadrivendiscovery.org/types/TuningParameter',
                    'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
                ),
                'description': 'Learning rate shrinks the contribution of each classifier by ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.',
                'lower': 0.01,
                'upper': 2,
                'lower_inclusive': True,
                'upper_inclusive': False,
            },
            'array1': {
                'type': hyperparams.Hyperparameter,
                'default': ((1, 2), (3, 4)),
                'structural_type': container.ndarray,
                'semantic_types': (
                    'https://metadata.datadrivendiscovery.org/types/TuningParameter',
                ),
            },
            'array2': {
                'type': hyperparams.Hyperparameter,
                'default': ((1, 2), (3, 4)),
                'structural_type': container.DataFrame,
                'semantic_types': (
                    'https://metadata.datadrivendiscovery.org/types/TuningParameter',
                ),
            },
        })

        json.dumps(TestPrimitive.metadata.to_json_structure())

    def test_package_validation(self):
        Inputs = container.List
        Outputs = container.List

        class Hyperparams(hyperparams.Hyperparams):
            pass

        with self.assertRaisesRegex(ValueError, 'Invalid package name'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'source': {
                            'name': 'Test',
                        },
                        'installation': [{
                            'type': metadata_base.PrimitiveInstallationType.PIP,
                            'package': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git',
                            'version': '0.1.0',
                        }],
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

    def test_package_uri_validation(self):
        Inputs = container.List
        Outputs = container.List

        class Hyperparams(hyperparams.Hyperparams):
            pass

        with self.assertRaisesRegex(ValueError, 'Package URI does not include a commit hash'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'source': {
                            'name': 'Test',
                        },
                        'installation': [{
                            'type': metadata_base.PrimitiveInstallationType.PIP,
                            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git',
                        }],
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

        with self.assertRaisesRegex(ValueError, 'Package URI does not include a commit hash'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'source': {
                            'name': 'Test',
                        },
                        'installation': [{
                            # Once with string.
                            'type': 'PIP',
                            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@v0.1.0',
                        }],
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

        with self.assertRaisesRegex(ValueError, 'Package URI does not include a commit hash'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'source': {
                            'name': 'Test',
                        },
                        'installation': [{
                            # Once with enum value.
                            'type': metadata_base.PrimitiveInstallationType.PIP,
                            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@v0.1.0',
                        }],
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

    def test_union_extra_argument(self):
        Inputs = typing.Union[container.List, container.ndarray]
        Outputs = container.List

        class Hyperparams(hyperparams.Hyperparams):
            pass

        # Silence any validation warnings.
        with utils.silence():
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': '5431cc97-9ebe-48c6-ae6d-e97a611e4a24',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, additional: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

                def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, additional: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                    return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, additional=additional)

                def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, additional: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                    return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, additional=additional)

    def test_subclass(self):
        Inputs = container.List
        Outputs = container.List

        class TestHyperparams(hyperparams.Hyperparams):
            a = hyperparams.Hyperparameter(
                default=0,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )

        # Silence any validation warnings.
        with utils.silence():
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, TestHyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': 'fd89a661-6aed-49ad-aa65-3d41ba9ee903',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

        class SubclassTestHyperparams(TestHyperparams):
            b = hyperparams.Hyperparameter(
                default=1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )

        # Silence any validation warnings.
        with utils.silence():
            class SubclassTestPrimitive(TestPrimitive, transformer.TransformerPrimitiveBase[Inputs, Outputs, SubclassTestHyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': 'f7ba1f51-ed06-4466-8fbd-857637b2d322',
                    'version': '0.1.0',
                    'name': "Subclass Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                })

        self.assertEqual(SubclassTestPrimitive.metadata.query()['id'], 'f7ba1f51-ed06-4466-8fbd-857637b2d322')
        self.assertIs(SubclassTestPrimitive.metadata.get_hyperparams(), SubclassTestHyperparams)
        self.assertEqual(set(SubclassTestHyperparams.configuration.keys()), {'a', 'b'})

        self.assertEqual(TestPrimitive.metadata.query()['id'], 'fd89a661-6aed-49ad-aa65-3d41ba9ee903')
        self.assertIs(TestPrimitive.metadata.get_hyperparams(), TestHyperparams)
        self.assertEqual(set(TestHyperparams.configuration.keys()), {'a'})

    def test_base_class_descriptions_constant(self):
        for loader, module_name, is_pkg in pkgutil.walk_packages(primitive_interfaces.__path__, primitive_interfaces.__name__ + '.'):
            if is_pkg:
                continue

            module = importlib.import_module(module_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if not issubclass(cls, base.PrimitiveBase):
                    continue

                # For each class that is a subclass of PrimitiveBase, check the doc string.
                self.assertTrue(cls.__doc__.startswith(base.DEFAULT_DESCRIPTION), '{module_name}.{name}'.format(module_name=module_name, name=name))


if __name__ == '__main__':
    unittest.main()
