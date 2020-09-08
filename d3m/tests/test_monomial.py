import json
import pickle
import unittest
import os.path
import sys

import d3m
from d3m import container, utils
from d3m.metadata import base

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.monomial import MonomialPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "4a0336ae-63b9-4a42-860e-86c5b64afbdd",
    "version": "0.1.0",
    "name": "Monomial Regressor",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "contact": "mailto:author@example.com",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py",
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
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/monomial.py"
    ],
    "python_path": "d3m.primitives.regression.monomial.Test",
    "algorithm_types": [
        "LINEAR_REGRESSION"
    ],
    "primitive_family": "REGRESSION",
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.monomial.MonomialPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Inputs": "d3m.container.list.List",
            "Outputs": "d3m.container.list.List",
            "Params": "test_primitives.monomial.Params",
            "Hyperparams": "test_primitives.monomial.Hyperparams"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "supervised_learning.SupervisedLearnerPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {
            "bias": {
                "type": "d3m.metadata.hyperparams.Hyperparameter",
                "default": 0.0,
                "structural_type": "float",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/TuningParameter"
                ]
            }
        },
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.monomial.Hyperparams",
                "kind": "RUNTIME"
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
            "random_seed": {
                "default": 0,
                "kind": "RUNTIME",
                "type": "int"
            },
            "inputs": {
                "type": "d3m.container.list.List",
                "kind": "PIPELINE"
            },
            "outputs": {
                "type": "d3m.container.list.List",
                "kind": "PIPELINE"
            },
            "params": {
                "type": "test_primitives.monomial.Params",
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
                "description": "Fits primitive using inputs and outputs (if any) using currently set training data.\n\nThe returned value should be a ``CallResult`` object with ``value`` set to ``None``.\n\nIf ``fit`` has already been called in the past on different training data,\nthis method fits it **again from scratch** using currently set training data.\n\nOn the other hand, caller can call ``fit`` multiple times on the same training data\nto continue fitting.\n\nIf ``fit`` fully fits using provided training data, there is no point in making further\ncalls to this method with same training data, and in fact further calls can be noops,\nor a primitive can decide to fully refit from scratch.\n\nIn the case fitting can continue with same training data (even if it is maybe not reasonable,\nbecause the internal metric primitive is using looks like fitting will be degrading), if ``fit``\nis called again (without setting training data), the primitive has to continue fitting.\n\nCaller can provide ``timeout`` information to guide the length of the fitting process.\nIdeally, a primitive should adapt its fitting process to try to do the best fitting possible\ninside the time allocated. If this is not possible and the primitive reaches the timeout\nbefore fitting, it should raise a ``TimeoutError`` exception to signal that fitting was\nunsuccessful in the given time. The state of the primitive after the exception should be\nas the method call has never happened and primitive should continue to operate normally.\nThe purpose of ``timeout`` is to give opportunity to a primitive to cleanly manage\nits state instead of interrupting execution from outside. Maintaining stable internal state\nshould have precedence over respecting the ``timeout`` (caller can terminate the misbehaving\nprimitive from outside anyway). If a longer ``timeout`` would produce different fitting,\nthen ``CallResult``'s ``has_finished`` should be set to ``False``.\n\nSome primitives have internal fitting iterations (for example, epochs). For those, caller\ncan provide how many of primitive's internal iterations should a primitive do before returning.\nPrimitives should make iterations as small as reasonable. If ``iterations`` is ``None``,\nthen there is no limit on how many iterations the primitive should do and primitive should\nchoose the best amount of iterations on its own (potentially controlled through\nhyper-parameters). If ``iterations`` is a number, a primitive has to do those number of\niterations (even if not reasonable), if possible. ``timeout`` should still be respected\nand potentially less iterations can be done because of that. Primitives with internal\niterations should make ``CallResult`` contain correct values.\n\nFor primitives which do not have internal iterations, any value of ``iterations``\nmeans that they should fit fully, respecting only ``timeout``.\n\nParameters\n----------\ntimeout:\n    A maximum time this primitive should be fitting during this method call, in seconds.\niterations:\n    How many of internal iterations should the primitive do.\n\nReturns\n-------\nA ``CallResult`` with ``None`` value."
            },
            "fit_multi_produce": {
                "kind": "OTHER",
                "arguments": [
                    "produce_methods",
                    "inputs",
                    "outputs",
                    "timeout",
                    "iterations"
                ],
                "returns": "d3m.primitive_interfaces.base.MultiCallResult",
                "description": "A method calling ``fit`` and after that multiple produce methods at once.\n\nThis method allows primitive author to implement an optimized version of both fitting\nand producing a primitive on same data.\n\nIf any additional method arguments are added to primitive's ``set_training_data`` method\nor produce method(s), or removed from them, they have to be added to or removed from this\nmethod as well. This method should accept an union of all arguments accepted by primitive's\n``set_training_data`` method and produce method(s) and then use them accordingly when\ncomputing results.\n\nThe default implementation of this method just calls first ``set_training_data`` method,\n``fit`` method, and all produce methods listed in ``produce_methods`` in order and is\npotentially inefficient.\n\nParameters\n----------\nproduce_methods:\n    A list of names of produce methods to call.\ninputs:\n    The inputs given to ``set_training_data`` and all produce methods.\noutputs:\n    The outputs given to ``set_training_data``.\ntimeout:\n    A maximum time this primitive should take to both fit the primitive and produce outputs\n    for all produce methods listed in ``produce_methods`` argument, in seconds.\niterations:\n    How many of internal iterations should the primitive do for both fitting and producing\n    outputs of all produce methods.\n\nReturns\n-------\nA dict of values for each produce method wrapped inside ``MultiCallResult``."
            },
            "get_params": {
                "kind": "OTHER",
                "arguments": [],
                "returns": "test_primitives.monomial.Params",
                "description": "Returns parameters of this primitive.\n\nParameters are all parameters of the primitive which can potentially change during a life-time of\na primitive. Parameters which cannot are passed through constructor.\n\nParameters should include all data which is necessary to create a new instance of this primitive\nbehaving exactly the same as this instance, when the new instance is created by passing the same\nparameters to the class constructor and calling ``set_params``.\n\nNo other arguments to the method are allowed (except for private arguments).\n\nReturns\n-------\nAn instance of parameters."
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
                "description": "Sets parameters of this primitive.\n\nParameters are all parameters of the primitive which can potentially change during a life-time of\na primitive. Parameters which cannot are passed through constructor.\n\nNo other arguments to the method are allowed (except for private arguments).\n\nParameters\n----------\nparams:\n    An instance of parameters."
            },
            "set_training_data": {
                "kind": "OTHER",
                "arguments": [
                    "inputs",
                    "outputs"
                ],
                "returns": "NoneType",
                "description": "Sets current training data of this primitive.\n\nThis marks training data as changed even if new training data is the same as\nprevious training data.\n\nStandard sublasses in this package do not adhere to the Liskov substitution principle when\ninheriting this method because they do not necessary accept all arguments found in the base\nclass. This means that one has to inspect which arguments are accepted at runtime, or in\nother words, one has to inspect which exactly subclass a primitive implements, if\nyou are accepting a wider range of primitives. This relaxation is allowed only for\nstandard subclasses found in this package. Primitives themselves should not break\nthe Liskov substitution principle but should inherit from a suitable base class.\n\nParameters\n----------\ninputs:\n    The inputs.\noutputs:\n    The outputs."
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
        },
        "params": {
            "a": "float"
        }
    },
    "structural_type": "test_primitives.monomial.MonomialPrimitive",
    "description": "A primitive which fits output = a * input.\n\nAttributes\n----------\nmetadata:\n    Primitive's metadata. Available as a class attribute.\nlogger:\n    Primitive's logger. Available as a class attribute.\nhyperparams:\n    Hyperparams passed to the constructor.\nrandom_seed:\n    Random seed passed to the constructor.\ndocker_containers:\n    A dict mapping Docker image keys from primitive's metadata to (named) tuples containing\n    container's address under which the container is accessible by the primitive, and a\n    dict mapping exposed ports to ports on that address.\nvolumes:\n    A dict mapping volume keys from primitive's metadata to file and directory paths\n    where downloaded and extracted files are available to the primitive.\ntemporary_directory:\n    An absolute path to a temporary directory a primitive can use to store any files\n    for the duration of the current pipeline run phase. Directory is automatically\n    cleaned up after the current pipeline run phase finishes.",
    "digest": "__DIGEST__"
}
""".replace('__INTERFACES_VERSION__', d3m.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR)).replace('__DIGEST__', MonomialPrimitive.metadata.query()['digest'])


class TestMonomialPrimitive(unittest.TestCase):
    def call_primitive(self, primitive, method_name, **kwargs):
        return getattr(primitive, method_name)(**kwargs)

    def test_basic(self):
        hyperparams_class = MonomialPrimitive.metadata.get_hyperparams()

        primitive = MonomialPrimitive(hyperparams=hyperparams_class.defaults())

        inputs = container.List([1, 2, 3, 4, 5, 6], generate_metadata=True)

        outputs = container.List([2, 4, 6, 8, 10, 12], generate_metadata=True)

        self.call_primitive(primitive, 'set_training_data', inputs=inputs, outputs=outputs)
        call_metadata = self.call_primitive(primitive, 'fit')

        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        inputs = container.List([10, 20, 30], generate_metadata=True)

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertSequenceEqual(call_metadata.value, [20, 40, 60])
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 3)
        self.assertEqual(call_metadata.value.metadata.query((base.ALL_ELEMENTS,))['structural_type'], float)

        call_metadata = primitive.multi_produce(produce_methods=('produce',), inputs=inputs)

        self.assertEqual(len(call_metadata.values), 1)
        self.assertSequenceEqual(call_metadata.values['produce'], [20, 40, 60])
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

    def test_hyperparameter(self):
        hyperparams_class = MonomialPrimitive.metadata.get_hyperparams()

        primitive = MonomialPrimitive(hyperparams=hyperparams_class(bias=1))

        inputs = container.List([1, 2, 3, 4, 5, 6], generate_metadata=True)

        outputs = container.List([2, 4, 6, 8, 10, 12], generate_metadata=True)

        self.call_primitive(primitive, 'set_training_data', inputs=inputs, outputs=outputs)
        call_metadata = self.call_primitive(primitive, 'fit')

        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        inputs = container.List([10, 20, 30], generate_metadata=True)

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertSequenceEqual(call_metadata.value, [21, 41, 61])
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 3)
        self.assertEqual(call_metadata.value.metadata.query((base.ALL_ELEMENTS,))['structural_type'], float)

    def test_recreation(self):
        hyperparams_class = MonomialPrimitive.metadata.get_hyperparams()

        primitive = MonomialPrimitive(hyperparams=hyperparams_class(bias=1))

        inputs = container.List([1, 2, 3, 4, 5, 6], generate_metadata=True)

        outputs = container.List([2, 4, 6, 8, 10, 12], generate_metadata=True)

        self.call_primitive(primitive, 'set_training_data', inputs=inputs, outputs=outputs)
        call_metadata = self.call_primitive(primitive, 'fit')

        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        params = self.call_primitive(primitive, 'get_params')

        pickled_params = pickle.dumps(params)
        unpickled_params = pickle.loads(pickled_params)

        self.assertEqual(params, unpickled_params)

        pickled_hyperparams = pickle.dumps(primitive.hyperparams)
        unpickled_hyperparams = pickle.loads(pickled_hyperparams)

        self.assertEqual(primitive.hyperparams, unpickled_hyperparams)

        primitive = MonomialPrimitive(hyperparams=unpickled_hyperparams)

        self.call_primitive(primitive, 'set_params', params=unpickled_params)

        inputs = container.List([10, 20, 30], generate_metadata=True)

        call_metadata =self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertSequenceEqual(call_metadata.value, [21, 41, 61])
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 3)
        self.assertEqual(call_metadata.value.metadata.query((base.ALL_ELEMENTS,))['structural_type'], float)

    def test_pickle(self):
        hyperparams_class = MonomialPrimitive.metadata.get_hyperparams()

        primitive = MonomialPrimitive(hyperparams=hyperparams_class(bias=1))

        inputs = container.List([1, 2, 3, 4, 5, 6], generate_metadata=True)

        outputs = container.List([2, 4, 6, 8, 10, 12], generate_metadata=True)

        self.call_primitive(primitive, 'set_training_data', inputs=inputs, outputs=outputs)
        call_metadata = self.call_primitive(primitive, 'fit')

        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        pickled_primitive = pickle.dumps(primitive)
        unpickled_primitive = pickle.loads(pickled_primitive)

        self.assertEqual(primitive.hyperparams, unpickled_primitive.hyperparams)
        self.assertEqual(primitive.random_seed, unpickled_primitive.random_seed)
        self.assertEqual(primitive.docker_containers, unpickled_primitive.docker_containers)

        inputs = container.List([10, 20, 30], generate_metadata=True)

        call_metadata =self.call_primitive(unpickled_primitive, 'produce', inputs=inputs)

        self.assertSequenceEqual(call_metadata.value, [21, 41, 61])
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 3)
        self.assertEqual(call_metadata.value.metadata.query((base.ALL_ELEMENTS,))['structural_type'], float)

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(MonomialPrimitive.metadata.to_json_structure()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
