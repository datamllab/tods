import os.path
import time
import typing

import numpy  # type: ignore

from d3m import container, exceptions, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__, null

__all__ = ('PrimitiveSumPrimitive',)

Inputs = container.List
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    # These primitives should already be fitted (or be a transformer) and they should accept
    # "List" container type as an input, and return a "List" container type as an output.
    # TODO: How to define this in the hyper-parameter definition?
    #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/210
    primitive_1 = hyperparams.Primitive[base.PrimitiveBase](
        default=null.NullTransformerPrimitive,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    primitive_2 = hyperparams.Primitive[base.PrimitiveBase](
        default=null.NullTransformerPrimitive,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class PrimitiveSumPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # It is important to provide a docstring because this docstring is used as a description of
    # a primitive. Some callers might analyze it to determine the nature and purpose of a primitive.

    """
    A primitive which element-wise sums the produced results of two other primitives. Each of those two primitives
    are given inputs (a list of numbers) to this primitive first as their inputs, are expected to return a list
    of numbers back, and then those lists are element-wise summed together, to produce the final list.

    This primitive exists just as a demonstration. To sum results you would otherwise just simply
    sum the results directly instead of getting an instance of the primitive and call
    produce methods on it. But this does allow more complicated ways of interacting with a
    primitive and this primitive demonstrates it.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '6b061902-5e40-4a7a-9a21-b995dce1b2aa',
        'version': __version__,
        'name': "Sum results of other primitives",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/primitive_sum.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # URIs at which one can obtain code for the primitive, if available.
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/add_primitives.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.operator.primitive_sum.Test',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
        # A metafeature about preconditions required for this primitive to operate well.
        'preconditions': [
            # Instead of strings you can also use available Python enumerations.
            metadata_base.PrimitivePrecondition.NO_MISSING_VALUES,
            metadata_base.PrimitivePrecondition.NO_CATEGORICAL_VALUES,
        ]
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        primitive_1 = self.hyperparams['primitive_1']
        primitive_2 = self.hyperparams['primitive_2']

        results = []

        if primitive_1 is not None:
            start = time.perf_counter()
            results.append(primitive_1.produce(inputs=inputs, timeout=timeout, iterations=iterations))
            delta = time.perf_counter() - start

            # Decrease the amount of time available to other calls. This delegates responsibility
            # of raising a "TimeoutError" exception to produce methods themselves. It also assumes
            # that if one passes a negative timeout value to a produce method, it raises a
            # "TimeoutError" exception correctly.
            if timeout is not None:
                timeout -= delta

        if primitive_2 is not None:
            results.append(primitive_2.produce(inputs=inputs, timeout=timeout, iterations=iterations))

        if not results:
            raise exceptions.InvalidArgumentValueError("No primitives provided as hyper-parameters.")

        # Even if the structure of outputs is the same as inputs, conceptually, outputs are different,
        # they are new data. So we do not reuse metadata from inputs but generate new metadata.
        outputs = container.List([sum(x) for x in zip(*[result.value for result in results])], generate_metadata=True)

        # We return the maximum number of iterations done by any produce method we called.
        iterations_done = None
        for result in results:
            if result.iterations_done is not None:
                if iterations_done is None:
                    iterations_done = result.iterations_done
                else:
                    iterations_done = max(iterations_done, result.iterations_done)

        return base.CallResult(
            value=outputs,
            has_finished=all(result.has_finished for result in results),
            iterations_done=iterations_done,
        )
