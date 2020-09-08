import json
import unittest
import os.path
import sys

import d3m
from d3m import container, utils
from d3m.metadata import base

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.primitive_sum import PrimitiveSumPrimitive
from test_primitives.null import NullTransformerPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "6b061902-5e40-4a7a-9a21-b995dce1b2aa",
    "version": "0.1.0",
    "name": "Sum results of other primitives",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "contact": "mailto:author@example.com",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/primitive_sum.py",
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
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/add_primitives.py"
    ],
    "python_path": "d3m.primitives.operator.primitive_sum.Test",
    "algorithm_types": [
        "COMPUTER_ALGEBRA"
    ],
    "primitive_family": "OPERATOR",
    "preconditions": [
        "NO_MISSING_VALUES",
        "NO_CATEGORICAL_VALUES"
    ],
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.primitive_sum.PrimitiveSumPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Inputs": "d3m.container.list.List",
            "Outputs": "d3m.container.list.List",
            "Hyperparams": "test_primitives.primitive_sum.Hyperparams",
            "Params": "NoneType"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "transformer.TransformerPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {
            "primitive_1": {
                "type": "d3m.metadata.hyperparams.Primitive",
                "default": "test_primitives.null.NullTransformerPrimitive",
                "structural_type": "d3m.primitive_interfaces.base.PrimitiveBase",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                "primitive_families": [

                ],
                "algorithm_types": [

                ],
                "produce_methods": [

                ]
            },
            "primitive_2": {
                "type": "d3m.metadata.hyperparams.Primitive",
                "default": "test_primitives.null.NullTransformerPrimitive",
                "structural_type": "d3m.primitive_interfaces.base.PrimitiveBase",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                "primitive_families": [

                ],
                "algorithm_types": [

                ],
                "produce_methods": [

                ]
            }
        },
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.primitive_sum.Hyperparams",
                "kind": "RUNTIME"
            },
            "random_seed": {
                "type": "int",
                "kind": "RUNTIME",
                "default":0
            },
            "docker_containers": {
                "type": "typing.Union[NoneType, typing.Dict[str, d3m.primitive_interfaces.base.DockerContainer]]",
                "kind": "RUNTIME",
                "default":null
            },
            "volumes": {
                "type": "typing.Union[NoneType, typing.Dict[str, str]]",
                "kind": "RUNTIME",
                "default":null
            },
            "temporary_directory": {
                "type": "typing.Union[NoneType, str]",
                "kind": "RUNTIME",
                "default": null
            },
            "timeout": {
                "type": "typing.Union[NoneType, float]",
                "kind": "RUNTIME",
                "default":null
            },
            "iterations": {
                "type": "typing.Union[NoneType, int]",
                "kind": "RUNTIME",
                "default":null
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
        "instance_methods": {
            "__init__": {
                "kind": "OTHER",
                "arguments": [
                    "hyperparams",
                    "random_seed",
                    "docker_containers",
                    "volumes",
                    "temporary_directory"
                ],
                "returns": "NoneType",
                "description": "All primitives should accept all their hyper-parameters in a constructor as one value,\nan instance of type ``Hyperparams``.\n\nProvided random seed should control all randomness used by this primitive.\nPrimitive should behave exactly the same for the same random seed across multiple\ninvocations. You can call `numpy.random.RandomState(random_seed)` to obtain an\ninstance of a random generator using provided seed. If your primitive does not\nuse randomness, consider not exposing this argument in your primitive's constructor\nto signal that.\n\nPrimitives can be wrappers around or use one or more Docker images which they can\nspecify as part of  ``installation`` field in their metadata. Each Docker image listed\nthere has a ``key`` field identifying that image. When primitive is created,\n``docker_containers`` contains a mapping between those keys and connection information\nwhich primitive can use to connect to a running Docker container for a particular Docker\nimage and its exposed ports. Docker containers might be long running and shared between\nmultiple instances of a primitive. If your primitive does not use Docker images,\nconsider not exposing this argument in your primitive's constructor.\n\n**Note**: Support for primitives using Docker containers has been put on hold.\nCurrently it is not expected that any runtime running primitives will run\nDocker containers for a primitive.\n\nPrimitives can also use additional static files which can be added as a dependency\nto ``installation`` metadata. When done so, given volumes are provided to the\nprimitive through ``volumes`` argument to the primitive's constructor as a\ndict mapping volume keys to file and directory paths where downloaded and\nextracted files are available to the primitive. All provided files and directories\nare read-only. If your primitive does not use static files, consider not exposing\nthis argument in your primitive's constructor.\n\nPrimitives can also use the provided temporary directory to store any files for\nthe duration of the current pipeline run phase. Directory is automatically\ncleaned up after the current pipeline run phase finishes. Do not store in this\ndirectory any primitive's state you would like to preserve between \"fit\" and\n\"produce\" phases of pipeline execution. Use ``Params`` for that. The main intent\nof this temporary directory is to store files referenced by any ``Dataset`` object\nyour primitive might create and followup primitives in the pipeline should have\naccess to. When storing files into this directory consider using capabilities\nof Python's `tempfile` module to generate filenames which will not conflict with\nany other files stored there. Use provided temporary directory as ``dir`` argument\nto set it as base directory to generate additional temporary files and directories\nas needed. If your primitive does not use temporary directory, consider not exposing\nthis argument in your primitive's constructor.\n\nNo other arguments to the constructor are allowed (except for private arguments)\nbecause we want instances of primitives to be created without a need for any other\nprior computation.\n\nModule in which a primitive is defined should be kept lightweight and on import not do\nany (pre)computation, data loading, or resource allocation/reservation. Any loading\nand resource allocation/reservation should be done in the constructor. Any (pre)computation\nshould be done lazily when needed once requested through other methods and not in the constructor."
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
                "description": "A method calling ``fit`` and after that multiple produce methods at once.\n\nParameters\n----------\nproduce_methods:\n    A list of names of produce methods to call.\ninputs:\n    The inputs given to all produce methods.\ntimeout:\n    A maximum time this primitive should take to both fit the primitive and produce outputs\n    for all produce methods listed in ``produce_methods`` argument, in seconds.\niterations:\n    How many of internal iterations should the primitive do for both fitting and producing\n    outputs of all produce methods.\n\nReturns\n-------\nA dict of values for each produce method wrapped inside ``MultiCallResult``."
            },
            "get_params": {
                "kind": "OTHER",
                "arguments": [

                ],
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
                "returns": "d3m.primitive_interfaces.base.CallResult[d3m.container.list.List]",
                "singleton":false,
                "inputs_across_samples": [

                ],
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
                "arguments": [

                ],
                "returns": "NoneType",
                "description": "A noop.\n\nParameters\n----------"
            }
        },
        "class_attributes": {
            "logger": "logging.Logger",
            "metadata": "d3m.metadata.base.PrimitiveMetadata"
        },
        "class_methods": {},
        "instance_attributes": {
            "hyperparams": "d3m.metadata.hyperparams.Hyperparams",
            "random_seed": "int",
            "docker_containers": "typing.Dict[str, d3m.primitive_interfaces.base.DockerContainer]",
            "volumes": "typing.Dict[str, str]",
            "temporary_directory": "typing.Union[NoneType, str]"
        }
    },
    "structural_type": "test_primitives.primitive_sum.PrimitiveSumPrimitive",
    "description": "A primitive which element-wise sums the produced results of two other primitives. Each of those two primitives\nare given inputs (a list of numbers) to this primitive first as their inputs, are expected to return a list\nof numbers back, and then those lists are element-wise summed together, to produce the final list.\n\nThis primitive exists just as a demonstration. To sum results you would otherwise just simply\nsum the results directly instead of getting an instance of the primitive and call\nproduce methods on it. But this does allow more complicated ways of interacting with a\nprimitive and this primitive demonstrates it.\n\nAttributes\n----------\nmetadata:\n    Primitive's metadata. Available as a class attribute.\nlogger:\n    Primitive's logger. Available as a class attribute.\nhyperparams:\n    Hyperparams passed to the constructor.\nrandom_seed:\n    Random seed passed to the constructor.\ndocker_containers:\n    A dict mapping Docker image keys from primitive's metadata to (named) tuples containing\n    container's address under which the container is accessible by the primitive, and a\n    dict mapping exposed ports to ports on that address.\nvolumes:\n    A dict mapping volume keys from primitive's metadata to file and directory paths\n    where downloaded and extracted files are available to the primitive.\ntemporary_directory:\n    An absolute path to a temporary directory a primitive can use to store any files\n    for the duration of the current pipeline run phase. Directory is automatically\n    cleaned up after the current pipeline run phase finishes.",
    "digest": "__DIGEST__"
}
""".replace('__INTERFACES_VERSION__', d3m.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR)).replace('__DIGEST__', PrimitiveSumPrimitive.metadata.query()['digest'])


class TestPrimitiveSumPrimitive(unittest.TestCase):
    def call_primitive(self, primitive, method_name, **kwargs):
        return getattr(primitive, method_name)(**kwargs)

    def test_basic(self):
        hyperparam_primitive1 = NullTransformerPrimitive(hyperparams=NullTransformerPrimitive.metadata.get_hyperparams().defaults())
        hyperparam_primitive2 = NullTransformerPrimitive(hyperparams=NullTransformerPrimitive.metadata.get_hyperparams().defaults())

        primitive = PrimitiveSumPrimitive(hyperparams={'primitive_1': hyperparam_primitive1, 'primitive_2': hyperparam_primitive2})
        inputs = container.List([10, 20, 30], generate_metadata=True)
        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertSequenceEqual(call_metadata.value, [20, 40, 60])
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 3)
        self.assertEqual(call_metadata.value.metadata.query((base.ALL_ELEMENTS,))['structural_type'], int)

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(PrimitiveSumPrimitive.metadata.to_json_structure()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
