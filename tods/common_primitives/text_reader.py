import os
from urllib import parse as url_parse

import frozendict  # type: ignore

from d3m import exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base

import common_primitives
from common_primitives import base


class TextReaderPrimitive(base.FileReaderPrimitiveBase):
    """
    A primitive which reads columns referencing plain text files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``text/plain``) has every filename read as a Python string. By default
    the resulting column with read strings is appended to existing columns.
    """

    _supported_media_types = (
        'text/plain',
    )
    _file_structural_type = str
    _file_semantic_types = ('http://schema.org/Text',)

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '0b21fcca-8b35-457d-a65d-36294c6f80a2',
            'version': '0.1.0',
            'name': 'Columns text reader',
            'python_path': 'd3m.primitives.data_preprocessing.text_reader.Common',
            'keywords': ['text', 'reader', 'plain'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/text_reader.py',
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
                metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
            ],
            'supported_media_types': _supported_media_types,
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        }
    )

    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> str:
        parsed_uri = url_parse.urlparse(fileuri, allow_fragments=False)

        if parsed_uri.scheme != 'file':
            raise exceptions.NotSupportedError("Only local files are supported, not '{fileuri}'.".format(fileuri=fileuri))

        if parsed_uri.netloc not in ['', 'localhost']:
            raise exceptions.InvalidArgumentValueError("Invalid hostname for a local file: {fileuri}".format(fileuri=fileuri))

        if not parsed_uri.path.startswith('/'):
            raise exceptions.InvalidArgumentValueError("Not an absolute path for a local file: {fileuri}".format(fileuri=fileuri))

        with open(parsed_uri.path, 'r', encoding='utf8') as file:
            return file.read()
