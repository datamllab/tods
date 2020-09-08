import os

import frozendict  # type: ignore
import imageio  # type: ignore
import numpy  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base

import common_primitives
from common_primitives import base


class DataFrameImageReaderPrimitive(base.FileReaderPrimitiveBase):
    """
    A primitive which reads columns referencing image files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``image/jpeg``, ``image/png``) has every filename read into an image
    represented as a numpy array. By default the resulting column with read arrays is appended
    to existing columns.

    The shape of numpy arrays is H x W x C. C is the number of channels in an image
    (e.g., C = 1 for greyscale, C = 3 for RGB), H is the height, and W is the width.
    dtype is uint8.
    """

    _supported_media_types = (
        'image/jpeg',
        'image/png',
    )
    _file_structural_type = container.ndarray
    _file_semantic_types = ('http://schema.org/ImageObject',)

    __author__ = 'University of Michigan, Ali Soltani'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8f2e51e8-da59-456d-ae29-53912b2b9f3d',
            'version': '0.2.0',
            'name': 'Columns image reader',
            'python_path': 'd3m.primitives.data_preprocessing.image_reader.Common',
            'keywords': ['image', 'reader', 'jpg', 'png'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:alsoltan@umich.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataframe_image_reader.py',
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

    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> container.ndarray:
        image_array = imageio.imread(fileuri)

        image_reader_metadata = image_array.meta

        # "imread" does not necessary always return uint8 dtype, but for PNG and JPEG files it should.
        assert image_array.dtype == numpy.uint8, image_array.dtype

        if image_array.ndim == 2:
            # Make sure there are always three dimensions.
            image_array = image_array.reshape(list(image_array.shape) + [1])

        assert image_array.ndim == 3, image_array.ndim

        image_array = container.ndarray(image_array, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        }, generate_metadata=False)

        # There might be custom metadata available, let's store it.
        # TODO: Add metadata which channel is which color (probably by providing metadata about the color space).
        #       It should probably go to "dimension" section for the "channels" dimension, for example, color space
        #       "RGB" would say that the dimension has to be of length 3 and has colors in this order.
        #       We could also set names for each dimension ("height", "width", "channels").
        #       We should probably also add semantic types to mark these dimensions.
        image_array.metadata = image_array.metadata.update((), {
            'image_reader_metadata': image_reader_metadata,
        })

        return image_array
