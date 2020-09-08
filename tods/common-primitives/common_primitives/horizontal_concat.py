import os
import typing

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('HorizontalConcatPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_index = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Use primary index columns in both DataFrames (if they exist) to match rows in proper order. Otherwise, concatination happens on the order of rows in input DataFrames.",
    )
    remove_second_index = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="When both input DataFrames have primary index columns, remove second index columns from the result."
                    " When \"use_index\" is \"True\", second index columns are redundant because they are equal to the first ones (assuming equal metadata).",
    )


class HorizontalConcatPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which concatenates two DataFrames horizontally.

    It has some heuristics how it tries to match up primary index columns in the case that there are
    multiple of them, but generally it aligns samples by all primary index columns.

    It is required that both DataFrames have the same number of samples.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'aff6a77a-faa0-41c5-9595-de2e7f7c4760',
            'version': '0.2.0',
            'name': "Concatenate two dataframes",
            'python_path': 'd3m.primitives.data_transformation.horizontal_concat.DataFrameCommon',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/horizontal_concat.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, left: Inputs, right: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # type: ignore
        return base.CallResult(left.horizontal_concat(
            right,
            use_index=self.hyperparams['use_index'],
            remove_second_index=self.hyperparams['remove_second_index'],
        ))

    def multi_produce(self, *, produce_methods: typing.Sequence[str], left: Inputs, right: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, left=left, right=right)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], left: Inputs, right: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, left=left, right=right)
