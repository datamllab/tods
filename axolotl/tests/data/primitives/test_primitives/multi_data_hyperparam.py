import os.path
import typing

import numpy as np  # type: ignore

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('MultiDataHyperparamPrimitive',)


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    values = hyperparams.Hyperparameter[typing.List[np.float64]](  # type: ignore
        default=[np.float64(1)],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The values to be added to input'
    )


class MultiDataHyperparamPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that requires a data argument hyperparam.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': 'ad8b8a35-9023-4f24-a628-a8f41eb2e3b0',
        'version': __version__,
        'name': "Multi Data Hyperparam Test Primitive",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/multi_data_hyperparam.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/multi_data_hyperparam.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.sum.MultiDataHyperparamTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        outputs = inputs
        for value in self.hyperparams['values']:
            outputs = outputs + value
        return base.CallResult(outputs)
