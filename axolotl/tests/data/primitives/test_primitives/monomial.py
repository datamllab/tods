import os.path
import typing

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, supervised_learning

from . import __author__, __version__

__all__ = ('MonomialPrimitive',)


# It is useful to define these names, so that you can reuse it both
# for class type arguments and method signatures.
Inputs = container.List
Outputs = container.List


class Params(params.Params):
    a: float


class Hyperparams(hyperparams.Hyperparams):
    bias = hyperparams.Hyperparameter(default=0.0, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])


class MonomialPrimitive(supervised_learning.SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    # It is important to provide a docstring because this docstring is used as a description of
    # a primitive. Some callers might analyze it to determine the nature and purpose of a primitive.

    """
    A primitive which fits output = a * input.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '4a0336ae-63b9-4a42-860e-86c5b64afbdd',
        'version': __version__,
        'name': "Monomial Regressor",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.regression.monomial.Test',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.LINEAR_REGRESSION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.REGRESSION,
    })

    # Random seed is not needed, but we need it in tests to test which random seed was passed to a primitive.
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._a: float = 0
        self._training_inputs: Inputs = None
        self._training_outputs: Outputs = None
        self._fitted: bool = False

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if self._a is None:
            raise ValueError("Calling produce before fitting.")

        # We compute the result. We use (...) here and not [...] to create a
        # generator and not a list which would then just be copied into "List".
        result = (self._a * input + self.hyperparams['bias'] for input in inputs)

        # We convert a regular list to container list which supports metadata attribute.
        # Even if the structure of outputs is the same as inputs, conceptually, outputs are different,
        # they are new data. So we do not reuse metadata from inputs but generate new metadata.
        outputs: container.List = container.List(result, generate_metadata=True)

        # Wrap it into default "CallResult" object: we are not doing any iterations.
        return base.CallResult(outputs)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        if not self._training_inputs or not self._training_inputs:
            raise ValueError("Missing training data.")

        quotients = [output / input for output, input in zip(self._training_outputs, self._training_inputs) if input != 0]
        self._a = sum(quotients) / len(quotients)
        self._fitted = True

        return base.CallResult(None)

    def get_params(self) -> Params:
        # You can pass a dict or keyword arguments.
        return Params(a=self._a)

    def set_params(self, *, params: Params) -> None:
        # Params are just a fancy dict.
        self._a = params['a']
