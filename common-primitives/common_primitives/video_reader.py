import os

import cv2  # type: ignore
import frozendict  # type: ignore
import numpy  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base

import common_primitives
from common_primitives import base


class VideoReaderPrimitive(base.FileReaderPrimitiveBase):
    """
    A primitive which reads columns referencing video files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``video/mp4``, ``video/avi``) has every filename read into a video
    represented as a numpy array. By default the resulting column with read arrays is appended
    to existing columns.

    The shape of numpy arrays is F x H x W x C. F is the number of frames, C is the number of
    channels in a video (e.g., C = 1 for greyscale, C = 3 for RGB), H is the height, and W
    is the width. dtype is uint8.
    """

    _supported_media_types = (
        'video/mp4',
        'video/avi',
    )
    _file_structural_type = container.ndarray
    _file_semantic_types = ('http://schema.org/VideoObject',)

    __author__ = 'University of Michigan, Eric Hofesmann, Nathan Louis, Madan Ravi Ganesh'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a29b0080-aeff-407d-9edb-0aa3eefbde01',
            'version': '0.2.0',
            'name': 'Columns video reader',
            'python_path': 'd3m.primitives.data_preprocessing.video_reader.Common',
            'keywords': ['video', 'reader', 'avi', 'mp4'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:davjoh@umich.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/video_reader.py',
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
        capture = cv2.VideoCapture(fileuri)
        frames = []

        try:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                else:
                    assert frame.dtype == numpy.uint8, frame.dtype

                    if frame.ndim == 2:
                        # Make sure there are always three dimensions.
                        frame = frame.reshape(list(frame.shape) + [1])

                    assert frame.ndim == 3, frame.ndim

                    frames.append(frame)
        finally:
            capture.release()

        return container.ndarray(numpy.array(frames), generate_metadata=False)
