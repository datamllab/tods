import os

import numpy  # type: ignore
import pandas  # type: ignore
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult

import common_primitives

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    separator = hyperparams.Hyperparameter[str](
        default='.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Separator separates additional identifier and original column name",
    )


class RenameDuplicateColumnsPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive renaming columns with duplicated name

    A numerical counter will be postfix on the original name and the original name will be stored in the other_name
    column metadata
    """

    __author__ = 'TAMU DARPA D3M Team, TsungLin Yang <lin.yang@tamu.edu>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7b067a78-4ad4-411d-9cf9-87bcee38ac73',
            'version': '0.2.0',
            'name': "Rename all the duplicated name column in DataFrame",
            'python_path': 'd3m.primitives.data_transformation.rename_duplicate_name.DataFrameCommon',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:lin.yang@tamu.edu',
                'uris': [
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
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        cols = pandas.Series(inputs.columns)
        dup_columns = inputs.columns[inputs.columns.duplicated()].unique()
        if not dup_columns.empty:
            inputs = inputs.copy()
            for dup in dup_columns:
                to_change_index = numpy.where(inputs.columns.values == dup)[0]
                new_names = [dup + self.hyperparams['separator'] + str(d_idx) if d_idx != 0 else dup for d_idx in
                             range(len(to_change_index))]
                for count, index in enumerate(to_change_index):
                    cols[index] = new_names[count]
                    inputs.metadata = inputs.metadata.update_column(index.item(), {'other_name': dup})
                    inputs.metadata = inputs.metadata.update_column(index.item(), {'name': cols[index]})
            inputs.columns = cols
        return CallResult(inputs)
