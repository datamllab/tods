import typing
import os
from urllib import parse as url_parse

import frozendict
import pandas

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.base import primitives



class CSVReaderPrimitive(primitives.FileReaderPrimitiveBase):    # pragma: no cover
    """
    A primitive which reads columns referencing CSV files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``text/csv``) has every filename read as a pandas DataFrame. By default
    the resulting column with read pandas DataFrames is appended to existing columns.
    """

    _supported_media_types = (
        'text/csv',
    )
    _file_structural_type = container.DataFrame
    _file_semantic_types = ('https://metadata.datadrivendiscovery.org/types/Table', 'https://metadata.datadrivendiscovery.org/types/Timeseries')

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '989562ac-b50f-4462-99cb-abef80d765b2',
            'version': '0.1.0',
            'name': 'Columns CSV reader',
            'python_path': 'd3m.primitives.tods.common.csv_reader',
            'keywords': ['CSV', 'reader'],
            'source': {
                'name': "DATALab@Texas A&M University",
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
            ],
            'supported_media_types': _supported_media_types,
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> container.DataFrame:
        # This is the same logic as used in D3M core package.
        # TODO: Deduplicate.

        expected_names = None
        if metadata.get('file_columns', None) is not None:
            expected_names = []
            for column in metadata['file_columns']:
                expected_names.append(column['column_name'])

        # Pandas requires a host for "file" URIs.
        parsed_uri = url_parse.urlparse(fileuri, allow_fragments=False)
        if parsed_uri.scheme == 'file' and parsed_uri.netloc == '':
            parsed_uri = parsed_uri._replace(netloc='localhost')
            fileuri = url_parse.urlunparse(parsed_uri)

        data = pandas.read_csv(
            fileuri,
            usecols=expected_names,
            # We do not want to do any conversion of values at this point.
            # This should be done by primitives later on.
            dtype=str,
            # We always expect one row header.
            header=None,
            # We want empty strings and not NaNs.
            na_filter=False,
            encoding='utf8',
            low_memory=False,
            memory_map=True,
        )

        column_names = list(data.columns)

        if expected_names is not None and expected_names != column_names:
            raise ValueError("Mismatch between column names in data {column_names} and expected names {expected_names}.".format(
                column_names=column_names,
                expected_names=expected_names,
            ))

        if data is None:
            raise FileNotFoundError("Data file for table '{file_path}' cannot be found.".format(
                file_path=fileuri,
            ))

        data = container.DataFrame(data, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
        }, generate_metadata=True)

        assert column_names is not None

        for i, column_name in enumerate(column_names):
            data.metadata = data.metadata.update((metadata_base.ALL_ELEMENTS, i), {
                'name': column_name,
                'structural_type': str,
            })
        return data

    def _produce_column_metadata(self, inputs_metadata: metadata_base.DataMetadata, column_index: int,
                                 read_files: typing.Sequence[typing.Any]) -> metadata_base.DataMetadata:
        # We do not pass "read_files" to parent method but we apply it at the end of this method ourselves.
        column_metadata = super()._produce_column_metadata(inputs_metadata, column_index, [])
        column_metadata = column_metadata.update_column(0, {
            # Clear metadata useful for filename columns.
            'file_columns': metadata_base.NO_VALUE,
        })

        # We might have metadata about columns, apply it here.
        column_meta = inputs_metadata.query_column(column_index)
        if column_meta.get('file_columns', None):
            for i, column in enumerate(column_meta['file_columns']):
                column_metadata = column_metadata.update((metadata_base.ALL_ELEMENTS, 0, metadata_base.ALL_ELEMENTS, i), column)

                # We know which columns are there, but also we know that we are reading everything as strings, so we can set that as well.
                column_metadata = column_metadata.update(
                    (metadata_base.ALL_ELEMENTS, 0, metadata_base.ALL_ELEMENTS, i),
                    {
                        'structural_type': str,
                        'column_name': metadata_base.NO_VALUE,
                        'column_index': metadata_base.NO_VALUE,
                    }
                )

        # A DataFrame is always a table as well.
        column_metadata = column_metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/Table')

        # We do not pass "read_files" to parent method but we apply it here ourselves.
        # This makes sure that metadata read from data override any metadata from metadata.
        for row_index, file in enumerate(read_files):
            column_metadata = file.metadata.copy_to(column_metadata, (), (row_index, 0))

        return column_metadata
