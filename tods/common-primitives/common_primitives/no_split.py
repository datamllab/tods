import os
import typing

import numpy  # type: ignore
import pandas  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams

import common_primitives
from common_primitives import base

__all__ = ('NoSplitDatasetSplitPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    pass


class NoSplitDatasetSplitPrimitive(base.TabularSplitPrimitiveBase[Hyperparams]):
    """
    A primitive which splits a tabular Dataset in a way that for all splits it
    produces the same (full) Dataset. Useful for unsupervised learning tasks. .
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '48c683ad-da9e-48cf-b3a0-7394dba5e5d2',
            'version': '0.1.0',
            'name': "No-split tabular dataset splits",
            'python_path': 'd3m.primitives.evaluation.no_split_dataset_split.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/no_split.py',
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
                metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION,
                metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        # We still go through the whole splitting process to assure full compatibility
        # (and error conditions) of a regular split, but we use all data for both splits.
        all_data = numpy.arange(len(attributes))

        return [(all_data, all_data)]
