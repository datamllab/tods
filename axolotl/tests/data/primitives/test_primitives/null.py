import os.path
import typing

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer, unsupervised_learning

from . import __author__, __version__

__all__ = ('NullTransformerPrimitive', 'NullUnsupervisedLearnerPrimitive', 'NullDataFrameUnsupervisedLearnerPrimitive')

Inputs = container.List
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    pass


class Params(params.Params):
    pass


class NullTransformerPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which passes through inputs as outputs.

    It does not really care if inputs is list.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': 'e0f83c35-fe3d-4fa6-92cf-f7421408eab5',
        'version': __version__,
        'name': "Produce the same as the input",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/null.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/add_primitives.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.null.TransformerTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        return base.CallResult(
            value=inputs
        )


class NullUnsupervisedLearnerPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which passes through inputs as outputs.

    It does not really care if inputs is list.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': '5310d7c4-89a0-4dab-8419-3285e650105a',
        'version': __version__,
        'name': "Produce the same as the input",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/null.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/add_primitives.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.null.UnsupervisedLearnerTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.

        Parameters
        ----------
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        A noop.
        """

        return base.CallResult(None)

    def get_params(self) -> Params:
        """
        A noop.
        """

        return Params()

    def set_params(self, *, params: Params) -> None:
        """
        A noop.
        """

        return

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        return base.CallResult(
            value=inputs
        )


DataframeInputs = container.DataFrame
DataframeOutputs = container.DataFrame


class NullDataFrameUnsupervisedLearnerPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[DataframeInputs, DataframeOutputs, Params, Hyperparams]):
    """
    A primitive which passes through inputs as outputs.

    It does not really care if inputs is a Dataframe.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': '0c063f7b-98d8-4d3c-91df-6a56623b9cc3',
        'version': __version__,
        'name': "Produce the same as the input",
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/null.py',
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
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/add_primitives.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.operator.null.DataFrameUnsupervisedLearnerTest',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
    })

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.

        Parameters
        ----------
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        A noop.
        """

        return base.CallResult(None)

    def get_params(self) -> Params:
        """
        A noop.
        """

        return Params()

    def set_params(self, *, params: Params) -> None:
        """
        A noop.
        """

        return

    def produce(self, *, inputs: DataframeInputs, timeout: float = None, iterations: int = None) -> base.CallResult[DataframeOutputs]:
        return base.CallResult(
            value=inputs
        )
