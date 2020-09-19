import os
import typing
import collections

import numpy  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base,  hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives
from common_primitives import dataset_utils

__all__ = ('DatasetSamplePrimitive',)

Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset


class Hyperparams(hyperparams.Hyperparams):
    starting_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="From which resource to start denormalizing. If \"None\" then it starts from the dataset entry point.",
    )
    sample_size = hyperparams.Union[typing.Union[int, float, None]](
        configuration=collections.OrderedDict(
            absolute=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=1,
                description='Sample an absolute number of rows from the dataset.',
            ),
            relative=hyperparams.Uniform(
                lower=0,
                upper=1,
                default=0.5,
                description='Sample a relative number of rows from the dataset.',
            ),
            all_rows=hyperparams.Constant(
                default=None,
                description='Sample all rows from the dataset',
            ),
        ),
        default='relative',
        description='Sample rows from the dataset according to either an absolute or relative value.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    replacement = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether to sample the data with replacement.",
    )
    delete_recursive = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Delete rows in other resources/tables which are not needed for rows left in the dataset entry point resource/table.",
    )


class DatasetSamplePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which samples the rows of a tabular Dataset.
    """

    __author__ = 'Distil'
    __version__ = '0.1.0'
    __contact__ = 'mailto:nklabs@newknowledge.com'

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '268315c1-7549-4aee-a4cc-28921cba74c0',
            'version': __version__,
            'name': "Dataset sampling primitive",
            'python_path': 'd3m.primitives.data_preprocessing.dataset_sample.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': __contact__,
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataset_sample.py',
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
                metadata_base.PrimitiveAlgorithmType.DATA_SPLITTING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        main_resource_id, main_resource = base_utils.get_tabular_resource(inputs, self.hyperparams['starting_resource'])

        # return inputs immediately if constant sample HP or number of rows to sample > number in dataset
        if self.hyperparams['sample_size'] is None or self.hyperparams['sample_size'] >= main_resource.shape[0]:
            return base.CallResult(inputs)

        # don't resample if we are working on test data
        target_columns = inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/TrueTarget'], at=(main_resource_id,))

        # only consider rows of input where target column is not missing
        row_indices_to_keep = set()
        row_indices_to_sample = set()
        for row_index in range(main_resource.shape[0]):
            row_target_values = main_resource.iloc[row_index, target_columns]
            # if there is any missing value in targets we assume is a test data row, or at least a row we should not sample
            if row_target_values.eq('').any() or row_target_values.isna().any():
                row_indices_to_keep.add(row_index)
            else:
                row_indices_to_sample.add(row_index)

        # generate random indices to sample
        local_random_state = numpy.random.RandomState(self.random_seed)

        if self.hyperparams['sample_size'] < 1:
            sample_rows = int(self.hyperparams['sample_size'] * len(row_indices_to_sample))
        else:
            sample_rows = self.hyperparams['sample_size']
        if sample_rows != 0 and len(row_indices_to_sample) != 0:
            # we sort row indices to be deterministic
            row_indices_to_keep.update(local_random_state.choice(sorted(row_indices_to_sample), size=sample_rows, replace=self.hyperparams['replacement']))

        output_dataset = dataset_utils.sample_rows(
            inputs,
            main_resource_id,
            row_indices_to_keep,
            inputs.get_relations_graph(),
            delete_recursive=self.hyperparams.get('delete_recursive', False),
        )

        return base.CallResult(output_dataset)
