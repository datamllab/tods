import os.path
import pickle
import typing
from http import client

import numpy  # type: ignore

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

from . import __author__, __version__

__all__ = ('SumPrimitive',)


DOCKER_KEY = 'summing'

# It is useful to define these names, so that you can reuse it both
# for class type arguments and method signatures.
# This is just an example of how to define a more complicated input type,
# which is in fact more restrictive than what the primitive can really handle.
# One could probably just use "typing.Union[typing.Container]" in this case, if accepting
# a wide range of input types.
Inputs = typing.Union[container.ndarray, container.DataFrame, container.List]
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    """
    No hyper-parameters for this primitive.
    """

    pass


class SumPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # It is important to provide a docstring because this docstring is used as a description of
    # a primitive. Some callers might analyze it to determine the nature and purpose of a primitive.

    """
    A primitive which sums all the values on input into one number.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '9c00d42d-382d-4177-a0e7-082da88a29c8',
        'version': __version__,
        'name': "Sum Values",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['test primitive'],
        'source': {
            'name': __author__,
            'contact': 'mailto:author@example.com',
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }, {
            'type': metadata_base.PrimitiveInstallationType.DOCKER,
            # A key under which information about a running container will be provided to the primitive.
            'key': DOCKER_KEY,
            'image_name': 'registry.gitlab.com/datadrivendiscovery/tests-data/summing',
            # Instead of a label, an exact hash of the image is required. This assures reproducibility.
            # You can see digests using "docker images --digests".
            'image_digest': 'sha256:f75e21720e44cfa29d8a8e239b5746c715aa7cf99f9fde7916623fabc30d3364',
        }],
        # URIs at which one can obtain code for the primitive, if available.
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/sum.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.operator.sum.Test',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.COMPUTER_ALGEBRA,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
        # A metafeature about preconditions required for this primitive to operate well.
        'preconditions': [
            # Instead of strings you can also use available Python enumerations.
            metadata_base.PrimitivePrecondition.NO_MISSING_VALUES,
            metadata_base.PrimitivePrecondition.NO_CATEGORICAL_VALUES,
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, docker_containers=docker_containers)

        # We cannot check for expected ports here because during class construction, a mock value is passed which has empty ports dict.
        if not self.docker_containers or DOCKER_KEY not in self.docker_containers:
            raise ValueError("Docker key '{docker_key}' missing among provided Docker containers.".format(docker_key=DOCKER_KEY))

    def _convert_value(self, value: typing.Any) -> typing.Union[numpy.ndarray, typing.List, typing.Any]:
        # Server does not know about container types, just standard numpy arrays and lists.
        if isinstance(value, container.ndarray):
            return value.view(numpy.ndarray)
        elif isinstance(value, container.List):
            return [self._convert_value(v) for v in value]
        else:
            return value

    @base.singleton
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # In the future, we should store here data in Arrow format into
        # Plasma store and just pass an ObjectId of data over HTTP.
        value = self._convert_value(inputs)
        data = pickle.dumps(value)

        # TODO: Retry if connection fails.
        #       This connection can sometimes fail because the service inside a Docker container
        #       is not yet ready, despite container itself already running. Primitive should retry
        #       a few times before aborting.

        # Primitive knows the port the container is listening on.
        connection = client.HTTPConnection(self.docker_containers[DOCKER_KEY].address, port=self.docker_containers[DOCKER_KEY].ports['8000/tcp'])
        # This simple primitive does not keep any state in the Docker container.
        # But if your primitive does have to associate requests with a primitive, consider
        # using Python's "id(self)" call to get an identifier of a primitive's instance.
        self.logger.debug("HTTP request: container=%(container)s", {'container': self.docker_containers[DOCKER_KEY]}, extra={'data': value})
        connection.request('POST', '/', data, {
            'Content-Type': 'multipart/form-data',
        })
        response = connection.getresponse()
        self.logger.debug("HTTP response: status=%(status)s", {'status': response.status}, extra={'response': response})

        if response.status != 200:
            raise ValueError("Invalid HTTP response status: {status}".format(status=response.status))

        result = float(response.read())

        # Outputs are different from inputs, so we do not reuse metadata from inputs but generate new metadata.
        outputs = container.List((result,), generate_metadata=True)

        # Wrap it into default "CallResult" object: we are not doing any iterations.
        return base.CallResult(outputs)
