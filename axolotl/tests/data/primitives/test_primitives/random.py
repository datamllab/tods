import os.path
import typing

import numpy  # type: ignore

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, generator

from . import __author__, __version__

__all__ = ('RandomPrimitive',)


Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    # These hyper-parameters can be both control or tuning parameter depending on their
    # role in a pipeline. So it depends how a pipeline is constructed: with them having
    # a fixed value or something which can be tuned. So they have two semantic types.
    mu = hyperparams.Hyperparameter[float](default=0.0, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])
    sigma = hyperparams.Hyperparameter[float](default=1.0, semantic_types=[
        'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ])


class RandomPrimitive(generator.GeneratorPrimitiveBase[Outputs, None, Hyperparams]):
    # It is important to provide a docstring because this docstring is used as a description of
    # a primitive. Some callers might analyze it to determine the nature and purpose of a primitive.

    """
    A primitive which draws random samples from a normal distribution.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'df3153a1-4411-47e2-bbc0-9d5e9925ad79',
        'version': __version__,
        'name': "Random Samples",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/random.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/random.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_generation.random.Test',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.MERSENNE_TWISTER,
            metadata_base.PrimitiveAlgorithmType.NORMAL_DISTRIBUTION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_GENERATION,
    })

    # It is not necessary to limit arguments this way, but we use it in tests to test that it is supported.
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

    def produce(self, *, inputs: container.List, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # We get as an input a list of non-negative integers, indices into the set of random values.
        # For each integer we redraw the number of samples up to that index at which time we return
        # the last value, the value for that index. We add one to the index because index can start
        # with 0 but we want to draw at least 1 number then.
        # TODO: Optimize this if the inputs are a sequence of integers, we could reuse the state.
        results = [numpy.random.RandomState(self.random_seed).normal(self.hyperparams['mu'], self.hyperparams['sigma'], i + 1)[-1] for i in inputs]

        # Outputs are different from inputs, so we do not reuse metadata from inputs but create new metadata.
        # We convert the list to a container DataFrame which supports metadata attribute.
        outputs = container.DataFrame({'results': results}, generate_metadata=True)

        # Wrap it into default "CallResult" object: we are not doing any iterations.
        return base.CallResult(outputs)

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        A noop.
        """

        return base.CallResult(None)

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: container.List, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # type: ignore
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        Parameters
        ----------
        produce_methods : Sequence[str]
            A list of names of produce methods to call.
        inputs : List
            The inputs given to all produce methods.
        timeout : float
            A maximum time this primitive should take to both fit the primitive and produce outputs
            for all produce methods listed in ``produce_methods`` argument, in seconds.
        iterations : int
            How many of internal iterations should the primitive do for both fitting and producing
            outputs of all produce methods.

        Returns
        -------
        MultiCallResult
            A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)  # type: ignore
