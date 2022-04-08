import os
import typing

import numpy
import pandas

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.base import primitives

from tods.utils import construct_primitive_metadata
__all__ = ('NoSplitDatasetSplitPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    pass


class NoSplitDatasetSplitPrimitive(primitives.TabularSplitPrimitiveBase[Hyperparams]):
    """
    A primitive which splits a tabular Dataset in a way that for all splits it
    produces the same (full) Dataset. Useful for unsupervised learning tasks. .
    """

    metadata = construct_primitive_metadata(module='detection_algorithm', name='no_split_dataset_split', id='48c683ad-da9e-48cf-b3a0-7394dba5e5d2', primitive_family='evaluation', algorithm= ['identity', 'data_split'], description='No-split tabular dataset splits')
    
    

    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        # We still go through the whole splitting process to assure full compatibility
        # (and error conditions) of a regular split, but we use all data for both splits.
        all_data = numpy.arange(len(attributes))

        return [(all_data, all_data)]
