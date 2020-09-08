import os.path
import typing

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, problem
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('FakeScorePrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class FakeScorePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that takes a DataFrame and returns hard-coded fake accuracy scores.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata(
        {
            'id': '1c4d5cbd-163c-424d-8be5-0f267641ae34',
            'version': __version__,
            'name': "Generate fake scores for testing",
            'source': {
                'name': __author__,
                'contact': 'mailto:author@example.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/fake_score.py',
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
                'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/fake_score.py'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            ],
            'python_path': 'd3m.primitives.evaluation.compute_scores.Test',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ACCURACY_SCORE,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def produce(  # type: ignore
            self, *, inputs: Inputs, score_dataset: container.Dataset, timeout: float = None,
            iterations: int = None,
    ) -> base.CallResult[Outputs]:
        outputs: typing.Dict[str, typing.List] = {
            'metric': [problem.PerformanceMetric.ACCURACY.name],
            'value': [1.0],
            'normalized': [1.0],
        }

        results = container.DataFrame(data=outputs, columns=list(outputs.keys()), generate_metadata=True)

        results.metadata = results.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
        )
        results.metadata = results.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            'https://metadata.datadrivendiscovery.org/types/Score',
        )
        results.metadata = results.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2),
            'https://metadata.datadrivendiscovery.org/types/Score',
        )

        return base.CallResult(results)

    def multi_produce(  # type: ignore
        self, *, produce_methods: typing.Sequence[str], inputs: Inputs,
        score_dataset: container.Dataset, timeout: float = None, iterations: int = None,
    ) -> base.MultiCallResult:
        return self._multi_produce(
            produce_methods=produce_methods, timeout=timeout, iterations=iterations,
            inputs=inputs, score_dataset=score_dataset,
        )

    def fit_multi_produce(  # type: ignore
        self, *, produce_methods: typing.Sequence[str], inputs: Inputs,
        score_dataset: container.Dataset, timeout: float = None, iterations: int = None
    ) -> base.MultiCallResult:
        return self._fit_multi_produce(
            produce_methods=produce_methods, timeout=timeout, iterations=iterations,
            inputs=inputs, score_dataset=score_dataset,
        )
