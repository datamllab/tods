import json
import unittest
import os.path
import pickle
import sys

import numpy

import d3m
from d3m import container, utils
from d3m.metadata import base

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.random import RandomPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "df3153a1-4411-47e2-bbc0-9d5e9925ad79",
    "version": "0.1.0",
    "name": "Random Samples",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "contact": "mailto:author@example.com",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/random.py",
            "https://gitlab.com/datadrivendiscovery/tests-data.git"
        ]
    },
    "installation": [
        {
            "type": "PIP",
            "package_uri": "git+https://gitlab.com/datadrivendiscovery/tests-data.git@__GIT_COMMIT__#egg=test_primitives&subdirectory=primitives"
        }
    ],
    "location_uris": [
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/random.py"
    ],
    "python_path": "d3m.primitives.data_generation.random.Test",
    "algorithm_types": [
        "MERSENNE_TWISTER",
        "NORMAL_DISTRIBUTION"
    ],
    "primitive_family": "DATA_GENERATION",
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.random.RandomPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Outputs": "d3m.container.pandas.DataFrame",
            "Params": "NoneType",
            "Hyperparams": "test_primitives.random.Hyperparams",
            "Inputs": "d3m.container.list.List"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "generator.GeneratorPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {
            "mu": {
                "type": "d3m.metadata.hyperparams.Hyperparameter",
                "default": 0.0,
                "structural_type": "float",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter",
                    "https://metadata.datadrivendiscovery.org/types/TuningParameter"
                ]
            },
            "sigma": {
                "type": "d3m.metadata.hyperparams.Hyperparameter",
                "default": 1.0,
                "structural_type": "float",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter",
                    "https://metadata.datadrivendiscovery.org/types/TuningParameter"
                ]
            }
        },
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.random.Hyperparams",
                "kind": "RUNTIME"
            },
            "random_seed": {
                "type": "int",
                "kind": "RUNTIME",
                "default": 0
            },
            "timeout": {
                "type": "typing.Union[NoneType, float]",
                "kind": "RUNTIME",
                "default": null
            },
            "iterations": {
                "type": "typing.Union[NoneType, int]",
                "kind": "RUNTIME",
                "default": null
            },
            "produce_methods": {
                "type": "typing.Sequence[str]",
                "kind": "RUNTIME"
            },
            "inputs": {
                "type": "d3m.container.list.List",
                "kind": "PIPELINE"
            },
            "params": {
                "type": "NoneType",
                "kind": "RUNTIME"
            }
        },
        "class_methods": {},
        "instance_methods": {
            "__init__": {
                "kind": "OTHER",
                "arguments": [
                    "hyperparams",
                    "random_seed"
                ],
                "returns": "NoneType"
            },
            "fit": {
                "kind": "OTHER",
                "arguments": [
                    "timeout",
                    "iterations"
                ],
                "returns": "d3m.primitive_interfaces.base.CallResult[NoneType]",
                "description": "A noop.\n\nParameters\n----------\ntimeout:\n    A maximum time this primitive should be fitting during this method call, in seconds.\niterations:\n    How many of internal iterations should the primitive do.\n\nReturns\n-------\nA ``CallResult`` with ``None`` value."
            },
            "fit_multi_produce": {
                "kind": "OTHER",
                "arguments": [
                    "produce_methods",
                    "inputs",
                    "timeout",
                    "iterations"
                ],
                "returns": "d3m.primitive_interfaces.base.MultiCallResult",
                "description": "A method calling ``fit`` and after that multiple produce methods at once.\n\nParameters\n----------\nproduce_methods : Sequence[str]\n    A list of names of produce methods to call.\ninputs : List\n    The inputs given to all produce methods.\ntimeout : float\n    A maximum time this primitive should take to both fit the primitive and produce outputs\n    for all produce methods listed in ``produce_methods`` argument, in seconds.\niterations : int\n    How many of internal iterations should the primitive do for both fitting and producing\n    outputs of all produce methods.\n\nReturns\n-------\nMultiCallResult\n    A dict of values for each produce method wrapped inside ``MultiCallResult``."
            },
            "get_params": {
                "kind": "OTHER",
                "arguments": [],
                "returns": "NoneType",
                "description": "A noop.\n\nReturns\n-------\nAn instance of parameters."
            },
            "multi_produce": {
                "kind": "OTHER",
                "arguments": [
                    "produce_methods",
                    "inputs",
                    "timeout",
                    "iterations"
                ],
                "returns": "d3m.primitive_interfaces.base.MultiCallResult",
                "description": "A method calling multiple produce methods at once.\n\nWhen a primitive has multiple produce methods it is common that they might compute the\nsame internal results for same inputs but return different representations of those results.\nIf caller is interested in multiple of those representations, calling multiple produce\nmethods might lead to recomputing same internal results multiple times. To address this,\nthis method allows primitive author to implement an optimized version which computes\ninternal results only once for multiple calls of produce methods, but return those different\nrepresentations.\n\nIf any additional method arguments are added to primitive's produce method(s), they have\nto be added to this method as well. This method should accept an union of all arguments\naccepted by primitive's produce method(s) and then use them accordingly when computing\nresults.\n\nThe default implementation of this method just calls all produce methods listed in\n``produce_methods`` in order and is potentially inefficient.\n\nIf primitive should have been fitted before calling this method, but it has not been,\nprimitive should raise a ``PrimitiveNotFittedError`` exception.\n\nParameters\n----------\nproduce_methods:\n    A list of names of produce methods to call.\ninputs:\n    The inputs given to all produce methods.\ntimeout:\n    A maximum time this primitive should take to produce outputs for all produce methods\n    listed in ``produce_methods`` argument, in seconds.\niterations:\n    How many of internal iterations should the primitive do.\n\nReturns\n-------\nA dict of values for each produce method wrapped inside ``MultiCallResult``."
            },
            "produce": {
                "kind": "PRODUCE",
                "arguments": [
                    "inputs",
                    "timeout",
                    "iterations"
                ],
                "returns": "d3m.primitive_interfaces.base.CallResult[d3m.container.pandas.DataFrame]",
                "singleton": false,
                "inputs_across_samples": [],
                "description": "Produce primitive's best choice of the output for each of the inputs.\n\nThe output value should be wrapped inside ``CallResult`` object before returning.\n\nIn many cases producing an output is a quick operation in comparison with ``fit``, but not\nall cases are like that. For example, a primitive can start a potentially long optimization\nprocess to compute outputs. ``timeout`` and ``iterations`` can serve as a way for a caller\nto guide the length of this process.\n\nIdeally, a primitive should adapt its call to try to produce the best outputs possible\ninside the time allocated. If this is not possible and the primitive reaches the timeout\nbefore producing outputs, it should raise a ``TimeoutError`` exception to signal that the\ncall was unsuccessful in the given time. The state of the primitive after the exception\nshould be as the method call has never happened and primitive should continue to operate\nnormally. The purpose of ``timeout`` is to give opportunity to a primitive to cleanly\nmanage its state instead of interrupting execution from outside. Maintaining stable internal\nstate should have precedence over respecting the ``timeout`` (caller can terminate the\nmisbehaving primitive from outside anyway). If a longer ``timeout`` would produce\ndifferent outputs, then ``CallResult``'s ``has_finished`` should be set to ``False``.\n\nSome primitives have internal iterations (for example, optimization iterations).\nFor those, caller can provide how many of primitive's internal iterations\nshould a primitive do before returning outputs. Primitives should make iterations as\nsmall as reasonable. If ``iterations`` is ``None``, then there is no limit on\nhow many iterations the primitive should do and primitive should choose the best amount\nof iterations on its own (potentially controlled through hyper-parameters).\nIf ``iterations`` is a number, a primitive has to do those number of iterations,\nif possible. ``timeout`` should still be respected and potentially less iterations\ncan be done because of that. Primitives with internal iterations should make\n``CallResult`` contain correct values.\n\nFor primitives which do not have internal iterations, any value of ``iterations``\nmeans that they should run fully, respecting only ``timeout``.\n\nIf primitive should have been fitted before calling this method, but it has not been,\nprimitive should raise a ``PrimitiveNotFittedError`` exception.\n\nParameters\n----------\ninputs:\n    The inputs of shape [num_inputs, ...].\ntimeout:\n    A maximum time this primitive should take to produce outputs during this method call, in seconds.\niterations:\n    How many of internal iterations should the primitive do.\n\nReturns\n-------\nThe outputs of shape [num_inputs, ...] wrapped inside ``CallResult``."
            },
            "set_params": {
                "kind": "OTHER",
                "arguments": [
                    "params"
                ],
                "returns": "NoneType",
                "description": "A noop.\n\nParameters\n----------\nparams:\n    An instance of parameters."
            },
            "set_training_data": {
                "kind": "OTHER",
                "arguments": [],
                "returns": "NoneType",
                "description": "A noop.\n\nParameters\n----------\noutputs:\n    The outputs."
            }
        },
        "class_attributes": {
            "logger": "logging.Logger",
            "metadata": "d3m.metadata.base.PrimitiveMetadata"
        },
        "instance_attributes": {
            "hyperparams": "d3m.metadata.hyperparams.Hyperparams",
            "random_seed": "int",
            "docker_containers": "typing.Dict[str, d3m.primitive_interfaces.base.DockerContainer]",
            "volumes": "typing.Dict[str, str]",
            "temporary_directory": "typing.Union[NoneType, str]"
        }
    },
    "structural_type": "test_primitives.random.RandomPrimitive",
    "description": "A primitive which draws random samples from a normal distribution.\n\nAttributes\n----------\nmetadata:\n    Primitive's metadata. Available as a class attribute.\nlogger:\n    Primitive's logger. Available as a class attribute.\nhyperparams:\n    Hyperparams passed to the constructor.\nrandom_seed:\n    Random seed passed to the constructor.\ndocker_containers:\n    A dict mapping Docker image keys from primitive's metadata to (named) tuples containing\n    container's address under which the container is accessible by the primitive, and a\n    dict mapping exposed ports to ports on that address.\nvolumes:\n    A dict mapping volume keys from primitive's metadata to file and directory paths\n    where downloaded and extracted files are available to the primitive.\ntemporary_directory:\n    An absolute path to a temporary directory a primitive can use to store any files\n    for the duration of the current pipeline run phase. Directory is automatically\n    cleaned up after the current pipeline run phase finishes.",
    "digest": "__DIGEST__"
}
""".replace('__INTERFACES_VERSION__', d3m.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR)).replace('__DIGEST__', RandomPrimitive.metadata.query()['digest'])


class TestRandomPrimitive(unittest.TestCase):
    def call_primitive(self, primitive, method_name, **kwargs):
        return getattr(primitive, method_name)(**kwargs)

    def test_basic(self):
        hyperparams_class = RandomPrimitive.metadata.get_hyperparams()

        primitive = RandomPrimitive(random_seed=42, hyperparams=hyperparams_class.defaults())

        inputs = container.List(list(range(4)), generate_metadata=True)

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.allclose(call_metadata.value.values, container.ndarray([0.496714153011, -0.138264301171, 0.647688538101, 1.52302985641]).reshape((4, 1))))
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query((base.ALL_ELEMENTS, 0))['structural_type'], numpy.float64)

    def test_pickle(self):
        # This test is not really useful anymore because primitive now does not keep random state
        # anymore but outputs depend only on inputs, and not on previous calls to "produce" method.

        hyperparams_class = RandomPrimitive.metadata.get_hyperparams()

        primitive = RandomPrimitive(random_seed=42, hyperparams=hyperparams_class.defaults())

        inputs = container.List(list(range(4)), generate_metadata=True)

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.allclose(call_metadata.value.values, container.ndarray([0.496714153011, -0.138264301171, 0.647688538101, 1.52302985641]).reshape(4, 1)))

        pickled_primitive = pickle.dumps(primitive)

        inputs = container.List(list(range(4, 8)), generate_metadata=True)

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.allclose(call_metadata.value.values, container.ndarray([-0.23415337, -0.23413696, 1.57921282, 0.76743473]).reshape(4, 1)))

        unpickled_primitive = pickle.loads(pickled_primitive)

        call_metadata = self.call_primitive(unpickled_primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.allclose(call_metadata.value.values, container.ndarray([-0.23415337, -0.23413696, 1.57921282, 0.76743473]).reshape(4, 1)))

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(RandomPrimitive.metadata.to_json_structure()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
