import os.path
import typing

import numpy as np  # type: ignore

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('ContainerHyperparamPrimitive',)


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    dataframe = hyperparams.Hyperparameter[container.DataFrame](
        default=container.DataFrame(0, index=np.arange(10), columns=['Values'], generate_metadata=True),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='The values to be added to input, element-wise'
    )


class ContainerHyperparamPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which uses a hyperparam of type container_argument.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': '442b600e-1144-11e9-ab14-d663bd873d93',
        'version': __version__,
        'name': "Container Hyperparam Tester",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/container_hyperparam.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/container_hyperparam.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.sum.ContainerHyperparamTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        outputs = inputs + self.hyperparams['dataframe']
        return base.CallResult(outputs)
