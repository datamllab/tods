import json
import unittest
import os
import os.path
import sys
import time

import docker
import numpy

import d3m
from d3m import container, utils
from d3m.metadata import base as metadata_base
from d3m.primitive_interfaces import base

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.sum import SumPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
    "version": "0.1.0",
    "name": "Sum Values",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "contact": "mailto:author@example.com",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py",
            "https://gitlab.com/datadrivendiscovery/tests-data.git"
        ]
    },
    "installation": [
        {
            "type": "PIP",
            "package_uri": "git+https://gitlab.com/datadrivendiscovery/tests-data.git@__GIT_COMMIT__#egg=test_primitives&subdirectory=primitives"
        },
        {
            "type": "DOCKER",
            "key": "summing",
            "image_name": "registry.gitlab.com/datadrivendiscovery/tests-data/summing",
            "image_digest": "sha256:f75e21720e44cfa29d8a8e239b5746c715aa7cf99f9fde7916623fabc30d3364"
        }
    ],
    "location_uris": [
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/sum.py"
    ],
    "python_path": "d3m.primitives.operator.sum.Test",
    "algorithm_types": [
        "COMPUTER_ALGEBRA"
    ],
    "primitive_family": "OPERATOR",
    "preconditions": [
        "NO_MISSING_VALUES",
        "NO_CATEGORICAL_VALUES"
    ],
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.sum.SumPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Inputs": "typing.Union[d3m.container.list.List, d3m.container.numpy.ndarray, d3m.container.pandas.DataFrame]",
            "Outputs": "d3m.container.list.List",
            "Hyperparams": "test_primitives.sum.Hyperparams",
            "Params": "NoneType"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "transformer.TransformerPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {},
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.sum.Hyperparams",
                "kind": "RUNTIME"
            },
            "docker_containers": {
                "type": "typing.Union[NoneType, typing.Dict[str, d3m.primitive_interfaces.base.DockerContainer]]",
                "kind": "RUNTIME",
                "default": null
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
                "type": "typing.Union[d3m.container.list.List, d3m.container.numpy.ndarray, d3m.container.pandas.DataFrame]",
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
                    "docker_containers"
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
                "description": "A method calling ``fit`` and after that multiple produce methods at once.\n\nParameters\n----------\nproduce_methods:\n    A list of names of produce methods to call.\ninputs:\n    The inputs given to all produce methods.\ntimeout:\n    A maximum time this primitive should take to both fit the primitive and produce outputs\n    for all produce methods listed in ``produce_methods`` argument, in seconds.\niterations:\n    How many of internal iterations should the primitive do for both fitting and producing\n    outputs of all produce methods.\n\nReturns\n-------\nA dict of values for each produce method wrapped inside ``MultiCallResult``."
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
                "returns": "d3m.primitive_interfaces.base.CallResult[d3m.container.list.List]",
                "singleton": true,
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
                "description": "A noop.\n\nParameters\n----------"
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
    "structural_type": "test_primitives.sum.SumPrimitive",
    "description": "A primitive which sums all the values on input into one number.\n\nAttributes\n----------\nmetadata:\n    Primitive's metadata. Available as a class attribute.\nlogger:\n    Primitive's logger. Available as a class attribute.\nhyperparams:\n    Hyperparams passed to the constructor.\nrandom_seed:\n    Random seed passed to the constructor.\ndocker_containers:\n    A dict mapping Docker image keys from primitive's metadata to (named) tuples containing\n    container's address under which the container is accessible by the primitive, and a\n    dict mapping exposed ports to ports on that address.\nvolumes:\n    A dict mapping volume keys from primitive's metadata to file and directory paths\n    where downloaded and extracted files are available to the primitive.\ntemporary_directory:\n    An absolute path to a temporary directory a primitive can use to store any files\n    for the duration of the current pipeline run phase. Directory is automatically\n    cleaned up after the current pipeline run phase finishes.",
    "digest": "__DIGEST__"
}
""".replace('__INTERFACES_VERSION__', d3m.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR)).replace('__DIGEST__', SumPrimitive.metadata.query()['digest'])


class TestSumPrimitive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docker_client = docker.from_env()

        cls.docker_containers = {}

        # Start all containers (this pulls images if they do not yet exist).
        installation = SumPrimitive.metadata.query().get('installation', [])
        for entry in installation:
            if entry['type'] != metadata_base.PrimitiveInstallationType.DOCKER:
                continue

            cls.docker_containers[entry['key']] = cls.docker_client.containers.run(
                '{image_name}@{image_digest}'.format(image_name=entry['image_name'], image_digest=entry['image_digest']),
                # Ports are mapped to random ports on the host so that they works in GitLab CI and Docker-in-Docker
                # environment (ports are mapped to the Docker-in-Docker container itself, not the real host).
                # In Docker-in-Docker environment you cannot directly connect to a container.
                detach=True, auto_remove=True, publish_all_ports=True,
            )

        # Wait a bit for things to run. Even if status is "running" it does
        # not really mean all services inside are really already running.
        time.sleep(5)  # 5 s

        # Wait for containers to be running.
        for container in cls.docker_containers.values():
            for _ in range(100):  # 100 * 100 ms = 10 s
                container.reload()
                if container.status == 'running':
                    assert container.attrs.get('NetworkSettings', {}).get('IPAddress', None)
                    break
                elif container.status in ('removing', 'paused', 'exited', 'dead'):
                    raise ValueError("Container '{container}' is not running.".format(container=container))

                time.sleep(0.1)  # 100 ms
            else:
                raise ValueError("Container '{container}' is not running.".format(container=container))

    @classmethod
    def tearDownClass(cls):
        for key, container in cls.docker_containers.items():
            container.stop()

        cls.docker_containers = {}

    def call_primitive(self, primitive, method_name, **kwargs):
        return getattr(primitive, method_name)(**kwargs)

    def _map_ports(self, ports):
        return {port: int(port_map[0]['HostPort']) for port, port_map in ports.items()}

    def get_docker_containers(self):
        if os.environ.get('GITLAB_CI', None):
            # In GitLab CI we use Docker-in-Docker to run containers, so container's ports are mapped to Docker-in-Docker
            # container itself (with hostname "docker") and not to the host.
            return {key: base.DockerContainer('docker', self._map_ports(container.attrs['NetworkSettings']['Ports'])) for key, container in self.docker_containers.items()}
        else:
            return {key: base.DockerContainer('localhost', self._map_ports(container.attrs['NetworkSettings']['Ports'])) for key, container in self.docker_containers.items()}

    def test_ndarray(self):
        with self.assertLogs(SumPrimitive.metadata.query()['python_path'], level='DEBUG') as cm:
            hyperparams_class = SumPrimitive.metadata.get_hyperparams()

            primitive = SumPrimitive(hyperparams=hyperparams_class.defaults(), docker_containers=self.get_docker_containers())

            inputs = container.ndarray([[1, 2, 3, 4], [5, 6, 7, 8]], generate_metadata=True)

            call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

            # Because it is a singleton produce method we can know that there is exactly one value in outputs.
            result = call_metadata.value[0]

            self.assertEqual(result, 36)
            self.assertEqual(call_metadata.has_finished, True)
            self.assertEqual(call_metadata.iterations_done, None)

            self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS,))['structural_type'], float)

        self.assertEqual(len(cm.records), 2)
        self.assertEqual(cm.records[0].name, SumPrimitive.metadata.query()['python_path'])
        self.assertEqual(cm.records[1].name, SumPrimitive.metadata.query()['python_path'])

        self.assertIsInstance(cm.records[0].data, numpy.ndarray)
        self.assertEqual(cm.records[1].response.status, 200)

    def test_lists(self):
        hyperparams_class = SumPrimitive.metadata.get_hyperparams()

        primitive = SumPrimitive(hyperparams=hyperparams_class.defaults(), docker_containers=self.get_docker_containers())

        inputs = container.List([container.List([1, 2, 3, 4]), container.List([5, 6, 7, 8])], generate_metadata=True)

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        # Because it is a singleton produce method we can know that there is exactly one value in outputs.
        result = call_metadata.value[0]

        self.assertEqual(result, 36)
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS,))['structural_type'], float)

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(SumPrimitive.metadata.to_json_structure()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
