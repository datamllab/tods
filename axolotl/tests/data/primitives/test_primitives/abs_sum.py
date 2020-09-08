import os.path
import typing

import numpy as np  # type: ignore

from d3m import container, utils, exceptions
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('AbsSumPrimitive',)


Inputs = typing.Union[container.ndarray, container.DataFrame, container.List]
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """

    pass


class AbsSumPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that sums the absolute value of the elements in a container and returns a list with a single value: the sum.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': '24de67db-aa08-4b66-85b2-b7be97154cf6',
        'version': __version__,
        'name': "Absolute Sum Test Primitive",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/abs_sum.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/abs_sum.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.sum.AbsTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    @base.singleton
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        result = np.abs(self._convert_value(inputs)).sum()
        outputs = container.List((result,), generate_metadata=True)
        return base.CallResult(outputs)

    def _convert_value(self, value: typing.Any) -> typing.Union[np.ndarray, typing.List, typing.Any]:
        if isinstance(value, container.ndarray):
            return value.view(np.ndarray)
        elif isinstance(value, container.List):
            return [self._convert_value(v) for v in value]
        elif isinstance(value, container.DataFrame):
            return value.values
        else:
            raise exceptions.InvalidArgumentTypeError('Input value must be an instance of `container.ndarray`, `container.List`, or `container.DataFrame.')
