import os.path
import typing

from d3m import container, exceptions, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('FailPrimitive',)


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):

    method_to_fail = hyperparams.Enumeration[str](
        values=['__init__', 'set_training_data', 'fit', 'produce', 'none'],
        default='produce',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The name of the method the user wants this primitive to fail on.",
    )


class IntentionalError(Exception):
    """
    Exception raised for testing purposes.

    Parameters
    ----------
    class_name : str
        Name of the class where the error occurred.
    method_name : str
        Name of the method where the error occurred.
    """

    def __init__(self, class_name: str, method_name: str) -> None:
        message = f"This is an exception raised by a(n) {class_name} object in the {method_name} method"
        super().__init__(message)


class FailPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which fails on the requested method (given as hyper-parameter).

    Moreover, primitive does not correctly preserve state so if you pickle
    and unpickle it, it does not seen itself as fitted anymore.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': 'd6dfbefa-0fb8-11e9-ab14-d663bd873d93',
        'version': __version__,
        'name': "Failure Tester",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/fail.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/fail.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.null.FailTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._conditional_fail('__init__')
        self._fitted = False

    def _conditional_fail(self, method_name: str) -> None:
        if self.hyperparams['method_to_fail'] == method_name:
            raise IntentionalError(self.__class__.__name__, method_name)

    def set_training_data(self) -> None:  # type: ignore
        self._conditional_fail('set_training_data')
        self._fitted = False
        super().set_training_data()

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        self._conditional_fail('fit')
        self._fitted = True
        return super().fit(timeout=timeout, iterations=iterations)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        self._conditional_fail('produce')
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive is not fitted.")
        return base.CallResult(inputs)
