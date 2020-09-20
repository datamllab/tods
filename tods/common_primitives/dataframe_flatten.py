from typing import List, Any
import os
import csv
import collections

import frozendict  # type: ignore
import pandas as pd  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.base import utils as base_utils

import common_primitives

__all__ = ('DataFrameFlattenPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not have any semantic type from \"from_semantic_types\", it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should the nested columns be appended, should they replace original columns, or should only the expanded columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    error_on_no_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no column is selected/provided. Otherwise issue a warning.",
    )


class DataFrameFlattenPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Cycles through the input dataframe and flattens the encountered nested structures (series & dataframes).
    Flattening involves creating a new row for each nested data row, and replicating the unnested row features.
    [
        a, b, [w, x],
        c, d, [y, z],
    ]

    yields:

    [
        a, b, w,
        a, b, x,
        c, d, y,
        c, d, z
    ]

    If the d3m index field is present and set as index, it will be updated to be multi index
    as needed. The primitive should be called after the referenced files have
    already been nested in the dataframe (using the CSVReader primitive for example).  The primitive can
    flatten mutiple nested columns, but is currently limited to supporting a nesting depth of 1.
    """

    __author__ = 'Uncharted Software',
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '1c4aed23-f3d3-4e6b-9710-009a9bc9b694',
            'version': '0.1.0',
            'name': 'DataFrame Flatten',
            'python_path': 'd3m.primitives.data_preprocessing.flatten.DataFrameCommon',
            'keywords': ['dataframe', 'flatten'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:chris.bethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataframe_flatten.py',
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
                metadata_base.PrimitiveAlgorithmType.DATA_DENORMALIZATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        }
    )

    def _expand_rows(self, inputs: Inputs, cols_to_expand: List[int], return_result: str, add_index_columns: bool) -> container.DataFrame:
        output_data = []

        # find the index columns and ignore that have nested contents (are flagged for expand)
        # Currently needed becaause the CSVReader seems to replicate the filename column metadata into
        # the expanded metadata column, causing the PrimaryKey type to show up in the original and nested
        index_columns = inputs.metadata.get_index_columns()
        index_columns = [col for col in index_columns if col not in cols_to_expand]

        # get the selectors for the metadata we need to copy
        metadata_sel: List[Any] = []
        for col_idx in range(len(inputs.columns)):
            if col_idx in cols_to_expand:
                expand_meta = inputs.metadata.query((metadata_base.ALL_ELEMENTS, col_idx, metadata_base.ALL_ELEMENTS))
                num_sub_cols = expand_meta['dimension']['length']
                for sub_col_idx in range(num_sub_cols):
                    metadata_sel.append((metadata_base.ALL_ELEMENTS, col_idx, metadata_base.ALL_ELEMENTS, sub_col_idx))
            elif return_result != 'new' or (return_result == 'new' and add_index_columns and col_idx in index_columns):
                metadata_sel.append((metadata_base.ALL_ELEMENTS, col_idx))

        # process every input row
        # the nested data will be a series containing a dataframe
        for t_row in inputs.itertuples(index=False, name=None):
            row_data = [t_row]

            # expand every nested column
            # every column to expand essentially becomes a cross product
            for col_index in cols_to_expand:
                # col_data is the expanded value for that nested column
                # row_data is the list of all expanded data for that row
                col_data = []
                for e_row in t_row[col_index].itertuples(index=False, name=None):
                    for s_row in row_data:
                        if return_result == 'new':
                            if add_index_columns:
                                data = [s_row[idx] for idx in index_columns]
                                col_data.append(data + list(e_row))
                            else:
                                col_data.append(e_row)
                        elif return_result == 'replace':
                            data = list(s_row)
                            [data.pop(idx) for idx in cols_to_expand]
                            col_data.append(data + list(e_row))
                        else:
                            raise ValueError(f"Unsupported return_result '{return_result}'")
                row_data = col_data
            output_data.extend(row_data)

        # wrap up as a dataframe and reset index now that merging is all done
        result = container.DataFrame(output_data, generate_metadata=True)
        for col_idx, col_metadata_selector in enumerate(metadata_sel):
            result.metadata = inputs.metadata.copy_to(result.metadata, col_metadata_selector, (metadata_base.ALL_ELEMENTS, col_idx))

        result.reset_index(inplace=True, drop=True)
        return result

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> List[int]:
        def _is_nested(col_index: int) -> bool:
            t = inputs_metadata.query((metadata_base.ALL_ELEMENTS, col_index))['structural_type']
            return issubclass(t, container.DataFrame)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            inputs_metadata,
            self.hyperparams['use_columns'],
            self.hyperparams['exclude_columns'],
            _is_nested,
        )

        # We are OK if no columns ended up being encoded.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.
        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can be encoded. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def produce(self, *,
                inputs: Inputs,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[Outputs]:

        container_dataframe = inputs

        to_expand_index = self._get_columns(inputs.metadata)
        if len(to_expand_index) > 0:
            inputs_clone = inputs.copy()
            container_dataframe = self._expand_rows(inputs_clone, to_expand_index, self.hyperparams['return_result'], self.hyperparams['add_index_columns'])
        else:
            if self.hyperparams['error_on_no_columns']:
                raise ValueError('No columns need flattening')
            else:
                self.logger.warning('No columns required flattening')

        # wrap as a D3M container
        return base.CallResult(container_dataframe)
