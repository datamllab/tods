import tempfile
import signal
import subprocess
import os

import frozendict  # type: ignore
import numpy  # type: ignore
import prctl  # type: ignore
from scipy.io import wavfile  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base

import common_primitives
from common_primitives import base


class AudioReaderPrimitive(base.FileReaderPrimitiveBase):
    """
    A primitive which reads columns referencing audio files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``audio/aiff``, ``audio/flac``, ``audio/ogg``, ``audio/wav``, ``audio/mpeg``)
    has every filename read into an audio represented as a numpy array. By default the resulting column
    with read arrays is appended to existing columns.

    The shape of numpy arrays is S x C. S is the number of samples, C is the number of
    channels in an audio (e.g., C = 1 for mono, C = 2 for stereo). dtype is float32.
    """

    _supported_media_types = (
        'audio/aiff',
        'audio/flac',
        'audio/ogg',
        'audio/wav',
        'audio/mpeg',
    )
    _file_structural_type = container.ndarray
    _file_semantic_types = ('http://schema.org/AudioObject',)

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '05e6eba3-2f5a-4934-8309-a6d17e099400',
            'version': '0.1.0',
            'name': 'Columns audio reader',
            'python_path': 'd3m.primitives.data_preprocessing.audio_reader.Common',
            'keywords': ['audio', 'reader', 'aiff', 'flac', 'ogg', 'wav', 'mpeg'],
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/audio_reader.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'build-essential',
                'version': '12.4ubuntu1',
            }, {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'libcap-dev',
                'version': '1:2.25-1.1',
            }, {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'ffmpeg',
                'version': '7:2.8.11-0',
            }, {
                # "python-prctl" requires "build-essential" and "libcap-dev". We list it here instead of
                # "setup.py" to not have to list these system dependencies for every common primitive (because
                # we cannot assure this primitive annotation gets installed first).
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'python-prctl',
                'version': '1.7',
            }, {
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
        # Ideally, temporary files are created in ramdisk by configuring Python's location of temporary files.
        with tempfile.NamedTemporaryFile(mode='rb') as output_file:
            # We use ffmpeg to convert all audio files to same format.
            args = [
                'ffmpeg',
                '-y',  # Always overwrite existing files.
                '-nostdin',  # No interaction.
                '-i', fileuri,  # Input file.
                '-vn',  # There is no video.
                '-acodec', 'pcm_f32le',  # We want everything in float32 dtype.
                '-f', 'wav',  # This will give us sample rate available in metadata.
                output_file.name,  # Output file.
            ]

            try:
                result = subprocess.run(
                    args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    # Setting "pdeathsig" will make the ffmpeg process be killed if our process dies for any reason.
                    encoding='utf8', check=True, preexec_fn=lambda: prctl.set_pdeathsig(signal.SIGKILL),
                )
            except subprocess.CalledProcessError as error:
                self.logger.error("Error running ffmpeg: %(stderr)s", {'stderr': error.stderr})
                raise

            self.logger.debug("Finished running ffmpeg: %(stderr)s", {'stderr': result.stderr})

            sampling_rate, audio_array = wavfile.read(output_file.name, mmap=True)

        assert audio_array.dtype == numpy.float32, audio_array.dtype

        if audio_array.ndim == 1:
            # Make sure there are always two dimensions.
            audio_array = audio_array.reshape(list(audio_array.shape) + [1])

        assert audio_array.ndim == 2, audio_array.ndim

        audio_array = container.ndarray(audio_array, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        }, generate_metadata=False)

        audio_array.metadata = audio_array.metadata.update((), {
            'dimension': {
                'sampling_rate': sampling_rate,
            },
        })

        return audio_array
