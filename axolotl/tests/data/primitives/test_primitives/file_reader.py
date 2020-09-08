import os
import typing

import numpy  # type: ignore
import frozendict  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.base import primitives
from d3m.metadata import base as metadata_base

from . import __author__, __version__

__all__ = ('DummyImageReaderPrimitive',)


class DummyImageReaderPrimitive(primitives.FileReaderPrimitiveBase):
    """
    A primitive which pretends to read columns referencing image files,
    but returns just the basename of the file path as dummy value of the file,
    wrapped inside a 1x1 ndarray.
    """

    _supported_media_types = (
        'image/jpeg',
        'image/png',
    )
    _file_structural_type = container.ndarray
    _file_semantic_types = ('http://schema.org/ImageObject',)

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata(
        {
            'id': '4f6e56b6-4ece-444b-9354-5a2b4e575a13',
            'version': __version__,
            'name': 'Dummy image reader',
            'python_path': 'd3m.primitives.data_preprocessing.image_reader.Test',
            'keywords': ['image', 'reader', 'jpg', 'png'],
            'source': {
                'name': __author__,
                'contact': 'mailto:author@example.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/file_reader.py',
                    'https://gitlab.com/datadrivendiscovery/tests-data.git',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
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

    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> container.ndarray:
        image_array = container.ndarray(numpy.array([[fileuri.split('/')[-1]]], dtype=object), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        }, generate_metadata=False)

        image_array.metadata = image_array.metadata.update((), {
            'image_reader_metadata': {
                'foobar': 42,
            },
        })

        return image_array
