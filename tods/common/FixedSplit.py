import os
import typing

import numpy
import pandas

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.base import primitives

__all__ = ('FixedSplitDatasetSplitPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    primary_index_values = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='A set of primary index values of the main resource belonging to the test (score) split. Cannot be set together with "row_indices".',
    )
    row_indices = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='A set of row indices of the main resource belonging to the test (score) split. Cannot be set together with "primary_index_values".',
    )
    delete_recursive = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Delete rows in other resources/tables which are not needed for rows left in the dataset entry point resource/table.",
    )


class FixedSplitDatasetSplitPrimitive(primitives.TabularSplitPrimitiveBase[Hyperparams]):
    """
    A primitive which splits a tabular Dataset in a way that uses for the test
    (score) split a fixed list of primary index values or row indices of the main
    resource to be used. All other rows are added used for the train split.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '1654f000-2178-4520-be4c-a95bc26b8d3a',
            'version': '0.1.0',
            'name': "Fixed split tabular dataset splits",
            'python_path': 'd3m.primitives.tods.evaluation.fixed_split_dataset_split',
            'source': {
                'name': "DATALab@TexasA&M University",
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/fixed_split.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        # This should be handled by "Set" hyper-parameter, but we check it here again just to be sure.
        if d3m_utils.has_duplicates(self.hyperparams['primary_index_values']):
            raise exceptions.InvalidArgumentValueError("\"primary_index_values\" hyper-parameter has duplicate values.")
        if d3m_utils.has_duplicates(self.hyperparams['row_indices']):
            raise exceptions.InvalidArgumentValueError("\"row_indices\" hyper-parameter has duplicate values.")

        if self.hyperparams['primary_index_values'] and self.hyperparams['row_indices']:
            raise exceptions.InvalidArgumentValueError("Both \"primary_index_values\" and \"row_indices\" cannot be provided.")

        if self.hyperparams['primary_index_values']:
            primary_index_values = numpy.array(self.hyperparams['primary_index_values'])

            index_columns = dataset.metadata.get_index_columns(at=(main_resource_id,))

            if not index_columns:
                raise exceptions.InvalidArgumentValueError("Cannot find index columns in the main resource of the dataset, but \"primary_index_values\" is provided.")

            main_resource = dataset[main_resource_id]
            # We reset the index so that the index corresponds to row indices.
            main_resource = main_resource.reset_index(drop=True)

            # We use just the "d3mIndex" column and ignore multi-key indices.
            # This works for now because it seems that every current multi-key
            # dataset in fact has an unique value in "d3mIndex" alone.
            # See: https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/issues/117
            index_column = index_columns[0]

            score_data = numpy.array(main_resource.loc[main_resource.iloc[:, index_column].isin(primary_index_values)].index)
            score_data_set = set(score_data)

            assert len(score_data) == len(score_data_set), (len(score_data), len(score_data_set))

            if len(score_data) != len(primary_index_values):
                raise exceptions.InvalidArgumentValueError("\"primary_index_values\" contains values which do not exist.")

        else:
            score_data = numpy.array(self.hyperparams['row_indices'])
            score_data_set = set(score_data)

            all_data_set = set(numpy.arange(len(attributes)))

            if not score_data_set <= all_data_set:
                raise exceptions.InvalidArgumentValueError("\"row_indices\" contains indices which do not exist, e.g., {indices}.".format(
                    indices=sorted(score_data_set - all_data_set)[:5],
                ))

        train_data = []
        for i in numpy.arange(len(attributes)):
            if i not in score_data_set:
                train_data.append(i)

        assert len(train_data) + len(score_data) == len(attributes), (len(train_data), len(score_data), len(attributes))

        return [(numpy.array(train_data), score_data)]
