import os
import random
import typing

from d3m import container, exceptions, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult, ContinueFitMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

from . import __author__, __version__

__all__ = ('RandomClassifierPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    classes: typing.Optional[typing.Sequence[typing.Any]]
    random_state: typing.Any


class Hyperparams(hyperparams.Hyperparams):
    pass


class RandomClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                                ContinueFitMixin[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive randomly classify a class. For test purposes.

    It uses the first column of ``outputs`` as a target column.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        'id': 'b8d0d982-fc53-4a3f-8a8c-a284fdd45bfd',
        'version': __version__,
        'name': "Random Classifier",
        'python_path': 'd3m.primitives.classification.random_classifier.Test',
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.BINARY_CLASSIFICATION,
            metadata_base.PrimitiveAlgorithmType.MULTICLASS_CLASSIFICATION
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/random_classifier.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/random_classifier.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._random: random.Random = random.Random()
        self._random.seed(random_seed)
        self._training_outputs: Outputs = None
        self._fitted = False
        self._classes: typing.List = []

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_outputs = outputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        self._classes = sorted(self._training_outputs.iloc[:, 0].unique().tolist())

        self._fitted = True

        return CallResult(None)

    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_outputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        _classes = self._training_outputs.iloc[:, 0].unique().tolist()
        self._classes = sorted(set(self._classes + _classes))

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Not fitted.")

        k = len(inputs)
        predictions = self._random.choices(self._classes, k=k)  # type: ignore

        result = container.DataFrame({'predictions': predictions}, generate_metadata=True)

        return CallResult(result)

    def get_params(self) -> Params:
        if self._fitted:
            return Params(
                classes=self._classes,
                random_state=self._random.getstate(),
            )
        else:
            return Params(
                classes=None,
                random_state=self._random.getstate(),
            )

    def set_params(self, *, params: Params) -> None:
        self._classes = params['classes']
        self._random.setstate(params['random_state'])
        if self._classes is not None:
            self._fitted = True
