import os.path
import typing

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('IncrementPrimitive',)


# It is useful to define these names, so that you can reuse it both
# for class type arguments and method signatures.
Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    # We can provide a type argument to a Hyperparameter class to signal which
    # structural type the Hyperparameter is. If you do not provide it, it is
    # automatically detected.
    # This is not a tuning parameter but a control parameter which should be decided
    # once during pipeline building but then fixed and not changed during hyper-parameter
    # tuning.
    amount = hyperparams.Hyperparameter[float](default=1, semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])


class IncrementPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # It is important to provide a docstring because this docstring is used as a description of
    # a primitive. Some callers might analyze it to determine the nature and purpose of a primitive.

    """
    A primitive which increments each value by a fixed amount, by default 1.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '5c9d5acf-7754-420f-a49f-90f4d9d0d694',
        'version': __version__,
        'name': "Increment Values",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/increment.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/increment.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.operator.increment.Test',
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

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # If "inputs" is container.DataFrame, then also result is.
        outputs = typing.cast(Outputs, inputs + float(self.hyperparams['amount']))

        # Metadata might not be preserved through operations, so we make sure and update metadata ourselves.
        # Because just values changed (but not structure) and the primitive is a transformation, we can reuse
        # inputs metadata, but generate new metadata for new value to assure everything is matching.
        outputs.metadata = inputs.metadata.generate(outputs)

        # Wrap it into default "CallResult" object: we are not doing any iterations.
        return base.CallResult(outputs)
