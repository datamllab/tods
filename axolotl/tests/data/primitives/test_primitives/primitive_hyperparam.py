import os.path
import typing

import pandas as pd  # type: ignore

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitive_interfaces import base, transformer
from test_primitives.increment import IncrementPrimitive, Hyperparams as IncrementPrimitiveHyperparams

from . import __author__, __version__

__all__ = ('PrimitiveHyperparamPrimitive',)


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    primitive = hyperparams.Hyperparameter[base.PrimitiveBase](
        default=IncrementPrimitive(hyperparams=IncrementPrimitiveHyperparams.defaults()),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The primitive instance to be passed to PrimitiveHyperparamPrimitive'
    )


class PrimitiveHyperparamPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that requires a data argument hyperparam.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': 'bd67f49a-bf10-4251-9774-019add57370b',
        'version': __version__,
        'name': "Primitive Hyperparam Test Primitive",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/primitive_hyperparam.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/primitive_hyperparam.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.sum.PrimitiveHyperparamTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        primitive = self.hyperparams['primitive']
        result = primitive.produce(inputs=inputs)
        data = result.value
        if isinstance(data, pd.DataFrame):
            value = data.iloc[0]
        else:
            value = data[0]
        outputs = inputs + value
        return base.CallResult(outputs)
