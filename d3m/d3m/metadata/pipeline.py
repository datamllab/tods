import abc
import argparse
import collections
import copy
import datetime
import json
import logging
import os
import os.path
import pprint
import sys
import traceback
import typing
import uuid

import dateparser  # type: ignore

from d3m import container, deprecate, environment_variables, exceptions, index, utils
from d3m.primitive_interfaces import base
from . import base as metadata_base, hyperparams as hyperparams_module

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import lib as pyarrow_lib  # type: ignore
except ModuleNotFoundError:
    pyarrow_lib = None

__all__ = (
    'Pipeline', 'Resolver', 'NoResolver', 'PrimitiveStep', 'SubpipelineStep', 'PlaceholderStep',
)

logger = logging.getLogger(__name__)

# Comma because we unpack the list of validators returned from "load_schema_validators".
PIPELINE_SCHEMA_VALIDATOR, = utils.load_schema_validators(metadata_base.SCHEMAS, ('pipeline.json',))

PIPELINE_SCHEMA_VERSION = 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json'

CONTROL_HYPERPARAMETER_SEMANTIC_TYPE = 'https://metadata.datadrivendiscovery.org/types/ControlParameter'


class TypeInfo(typing.NamedTuple):
    structural_type: type
    singleton: typing.Optional[bool]


class Resolver:
    """
    A resolver to resolve primitives and pipelines.

    It resolves primitives from available primitives on the system,
    and resolves pipelines from files in pipeline search paths.

    Attributes
    ----------
    strict_resolving:
        If resolved pipeline or primitive does not fully match specified primitive reference, raise an exception?
    strict_digest:
        When loading pipelines or primitives, if computed digest does not match the one provided in metadata, raise an exception?
    pipeline_search_paths:
        A list of paths to directories with pipelines to resolve from.
        Their files should be named ``<pipeline id>.json``, ``<pipeline id>.yml``, or ``<pipeline id>.yaml``.

    Parameters
    ----------
    strict_resolving:
        If resolved pipeline or primitive does not fully match specified primitive reference, raise an exception?
    strict_digest:
        When loading pipelines or primitives, if computed digest does not match the one provided in metadata, raise an exception?
    pipeline_search_paths:
        A list of paths to directories with pipelines to resolve from.
        Their files should be named ``<pipeline id>.json``, ``<pipeline id>.yml``, or ``<pipeline id>.yaml``.
    respect_environment_variable:
        Use also (colon separated) pipeline search paths from ``PIPELINES_PATH`` environment variable?
    load_all_primitives:
        Load all primitives before attempting to resolve them. If ``False`` any primitive used in a
        pipeline has to be loaded before calling the resolver.
    primitives_blocklist:
        A collection of primitive path prefixes to not (try to) load.
    """

    strict_resolving: bool
    strict_digest: bool
    pipeline_search_paths: typing.Sequence[str]

    def __init__(self, *, strict_resolving: bool = False, strict_digest: bool = False,
                 pipeline_search_paths: typing.Sequence[str] = None,
                 respect_environment_variable: bool = True, load_all_primitives: bool = True,
                 primitives_blocklist: typing.Collection[str] = None) -> None:
        self.strict_resolving = strict_resolving
        self.strict_digest = strict_digest
        self.primitives_blocklist = primitives_blocklist

        if pipeline_search_paths is None:
            self.pipeline_search_paths: typing.List[str] = []
        else:
            self.pipeline_search_paths = typing.cast(typing.List[str], pipeline_search_paths)

        if respect_environment_variable:
            self.pipeline_search_paths += [path for path in os.environ.get(environment_variables.PIPELINES_PATH, '').split(':') if path]

        self._load_all_primitives = load_all_primitives
        self._primitives_loaded = False
        self._get_primitive_failed: typing.Set[str] = set()

    def get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        primitive = self._get_primitive(primitive_description)

        # This class always resolves a primitive, or throws an exception, but subclasses might return "None".
        if primitive is not None:
            self._check_primitive(primitive_description, primitive)

        return primitive

    @classmethod
    def get_pipeline_class(cls) -> 'typing.Type[Pipeline]':
        return Pipeline

    def get_pipeline(self, pipeline_description: typing.Dict) -> 'typing.Optional[Pipeline]':
        pipeline = self._get_pipeline(pipeline_description)

        # This class always resolves a pipeline, or throws an exception, but subclasses might return "None".
        if pipeline is not None:
            self._check_pipeline(pipeline_description, pipeline)

        return pipeline

    def _get_pipeline(self, pipeline_description: typing.Dict) -> 'typing.Optional[Pipeline]':
        # If more than just "id" and "digest" is in the pipeline description,
        # then we assume it is a full pipeline description. Digest is optional.
        if set(pipeline_description.keys()) - {'digest'} > {'id'}:
            return self._from_structure(pipeline_description)
        else:
            return self._from_file(pipeline_description)

    def _from_structure(self, pipeline_description: typing.Dict) -> 'typing.Optional[Pipeline]':
        return self.get_pipeline_class().from_json_structure(pipeline_description, resolver=self, strict_digest=self.strict_digest)

    def _from_file(self, pipeline_description: typing.Dict) -> 'typing.Optional[Pipeline]':
        for path in self.pipeline_search_paths:
            for extension in ['json', 'json.gz']:
                pipeline_path = os.path.join(path, '{pipeline_id}.{extension}'.format(pipeline_id=pipeline_description['id'], extension=extension))
                try:
                    with utils.open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
                        return self.get_pipeline_class().from_json(pipeline_file, resolver=self, strict_digest=self.strict_digest)
                except FileNotFoundError:
                    pass

            for extension in ['yml', 'yaml', 'yml.gz', 'yaml.gz']:
                pipeline_path = os.path.join(path, '{pipeline_id}.{extension}'.format(pipeline_id=pipeline_description['id'], extension=extension))
                try:
                    with utils.open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
                        return self.get_pipeline_class().from_yaml(pipeline_file, resolver=self, strict_digest=self.strict_digest)
                except FileNotFoundError:
                    pass

        raise exceptions.InvalidArgumentValueError("Unable to get pipeline '{pipeline_id}'.".format(pipeline_id=pipeline_description['id']))

    def _get_primitive_by_path(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        if primitive_description['python_path'] in self._get_primitive_failed:
            return None

        try:
            # We first try to directly load the primitive using its Python path.
            primitive = index.get_primitive(primitive_description['python_path'])
        except Exception:
            # We make sure we attempt to directly load the primitive only once. Otherwise error messages
            # during loading could be printed out again and again, every time we try to get this primitive.
            self._get_primitive_failed.add(primitive_description['python_path'])
            primitive = None

        # Then we check that the loaded primitive matches the requested primitive ID.
        # This way we can load primitive's without having to load all primitives in
        # the common case, when the Python path of the primitive has not changed.
        if primitive is not None and primitive.metadata.query()['id'] == primitive_description['id']:
            return primitive

        return None

    def _load_primitives(self) -> None:
        if not self._load_all_primitives:
            return

        if self._primitives_loaded:
            return
        self._primitives_loaded = True

        # We attempt to load all primitives only once. Otherwise error messages for failed primitives
        # during loading could be printed out again and again.
        index.load_all(blocklist=self.primitives_blocklist)

    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        if not self._primitives_loaded:
            primitive = self._get_primitive_by_path(primitive_description)

            if primitive is not None:
                return primitive

        self._load_primitives()

        return index.get_primitive_by_id(primitive_description['id'])

    def _check_primitive(self, primitive_description: typing.Dict, primitive: typing.Type[base.PrimitiveBase]) -> None:
        primitive_metadata = primitive.metadata.query()

        if primitive_metadata['version'] != primitive_description['version']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "Version for primitive '{primitive_id}' does not match the one specified in the primitive description. "
                    "Primitive description version: '{primitive_version}'. Resolved primitive version: '{resolved_primitive_version}'.".format(
                        primitive_id=primitive_metadata['id'],
                        primitive_version=primitive_description['version'],
                        resolved_primitive_version=primitive_metadata['version'],
                    )
                )
            else:
                logger.warning(
                    "Version for primitive '%(primitive_id)s' does not match the one specified in the primitive description. "
                    "Primitive description version: '%(primitive_version)s'. Resolved primitive version: '%(resolved_primitive_version)s'.",
                    {
                        'primitive_id': primitive_metadata['id'],
                        'primitive_version': primitive_description['version'],
                        'resolved_primitive_version': primitive_metadata['version'],
                    },
                )

        if primitive_metadata['python_path'] != primitive_description['python_path']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "Python path for primitive '{primitive_id}' does not match the one specified in the primitive description. "
                    "Primitive description Python path: '{primitive_python_path}'. Resolved primitive Python path: '{resolved_primitive_python_path}'.".format(
                        primitive_id=primitive_metadata['id'],
                        primitive_python_path=primitive_description['python_path'],
                        resolved_primitive_python_path=primitive_metadata['python_path'],
                    )
                )
            else:
                logger.warning(
                    "Python path for primitive '%(primitive_id)s' does not match the one specified in the primitive description. "
                    "Primitive description Python path: '%(primitive_python_path)s'. Resolved primitive Python path: '%(resolved_primitive_python_path)s'.",
                    {
                        'primitive_id': primitive_metadata['id'],
                        'primitive_python_path': primitive_description['python_path'],
                        'resolved_primitive_python_path': primitive_metadata['python_path'],
                    },
                )

        if primitive_metadata['name'] != primitive_description['name']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "Name for primitive '{primitive_id}' does not match the one specified in the primitive description. "
                    "Primitive description name: '{primitive_name}'. Resolved primitive name: '{resolved_primitive_name}'.".format(
                        primitive_id=primitive_metadata['id'],
                        primitive_name=primitive_description['name'],
                        resolved_primitive_name=primitive_metadata['name'],
                    )
                )
            else:
                logger.warning(
                    "Name for primitive '%(primitive_id)s' does not match the one specified in the primitive description. "
                    "Primitive description name: '%(primitive_name)s'. Resolved primitive name: '%(resolved_primitive_name)s'.",
                    {
                        'primitive_id': primitive_metadata['id'],
                        'primitive_name': primitive_description['name'],
                        'resolved_primitive_name': primitive_metadata['name'],
                    },
                )

        if 'digest' in primitive_description:
            assert primitive_description['digest'] is not None

            if primitive_metadata.get('digest', None) != primitive_description['digest']:
                if self.strict_digest:
                    raise exceptions.DigestMismatchError(
                        "Digest for primitive '{primitive_id}' does not match the one specified in the primitive description. "
                        "Primitive description digest: {primitive_digest}. Resolved primitive digest: {resolved_primitive_digest}.".format(
                            primitive_id=primitive_metadata['id'],
                            primitive_digest=primitive_description['digest'],
                            resolved_primitive_digest=primitive_metadata.get('digest', None),
                        )
                    )
                else:
                    logger.warning(
                        "Digest for primitive '%(primitive_id)s' does not match the one specified in the primitive description. "
                        "Primitive description digest: %(primitive_digest)s. Resolved primitive digest: %(resolved_primitive_digest)s.",
                        {
                            'primitive_id': primitive_metadata['id'],
                            'primitive_digest': primitive_description['digest'],
                            'resolved_primitive_digest': primitive_metadata.get('digest', None),
                        },
                    )

    def _check_pipeline(self, pipeline_description: typing.Dict, pipeline: 'Pipeline') -> None:
        # This can happen if the file has a filename for one pipeline ID but the contents have another pipeline ID.
        if pipeline.id != pipeline_description['id']:
            if self.strict_resolving:
                raise exceptions.MismatchError(
                    "ID of pipeline '{resolved_pipeline_id}' does not match the one specified in the pipeline description. "
                    "Pipeline description ID: '{pipeline_id}'. Resolved pipeline ID: '{resolved_pipeline_id}'.".format(
                        pipeline_id=pipeline_description['id'],
                        resolved_pipeline_id=pipeline.id,
                    )
                )
            else:
                logger.warning(
                    "ID of pipeline '%(resolved_pipeline_id)s' does not match the one specified in the pipeline description. "
                    "Pipeline description ID: '%(pipeline_id)s'. Resolved pipeline ID: '%(resolved_pipeline_id)s'.",
                    {
                        'pipeline_id': pipeline_description['id'],
                        'resolved_pipeline_id': pipeline.id,
                    },
                )

        if 'digest' in pipeline_description:
            assert pipeline_description['digest'] is not None

            pipeline_digest = pipeline.get_digest()

            if pipeline_digest != pipeline_description['digest']:
                if self.strict_digest:
                    raise exceptions.DigestMismatchError(
                        "Digest for pipeline '{pipeline_id}' does not match the one specified in the pipeline description. "
                        "Pipeline description digest: {pipeline_digest}. Resolved pipeline digest: {resolved_pipeline_digest}.".format(
                            pipeline_id=pipeline.id,
                            pipeline_digest=pipeline_description['digest'],
                            resolved_pipeline_digest=pipeline_digest,
                        )
                    )
                else:
                    logger.warning(
                        "Digest for pipeline '%(pipeline_id)s' does not match the one specified in the pipeline description. "
                        "Pipeline description digest: %(pipeline_digest)s. Resolved pipeline digest: %(resolved_pipeline_digest)s.",
                        {
                            'pipeline_id': pipeline.id,
                            'pipeline_digest': pipeline_description['digest'],
                            'resolved_pipeline_digest': pipeline_digest,
                        },
                    )


class NoResolver(Resolver):
    """
    A resolver which never resolves anything.
    """

    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        return None

    def _get_pipeline(self, pipeline_description: typing.Dict) -> 'typing.Optional[Pipeline]':
        return None


S = typing.TypeVar('S', bound='StepBase')


class StepBase(metaclass=utils.AbstractMetaclass):
    """
    Class representing one step in pipeline's execution.

    Attributes
    ----------
    index:
        An index of the step among steps in the pipeline.
    resolver:
        Resolver to use.

    Parameters
    ----------
    resolver:
        Resolver to use.
    """

    index: int
    resolver: Resolver

    def __init__(self, *, resolver: typing.Optional[Resolver] = None) -> None:
        self.resolver = self.get_resolver(resolver)

        self.index: int = None

    @classmethod
    def get_resolver(cls, resolver: typing.Optional[Resolver]) -> Resolver:
        if resolver is None:
            return Resolver()
        else:
            return resolver

    @classmethod
    @abc.abstractmethod
    def get_step_type(cls) -> metadata_base.PipelineStepType:
        pass

    def check_add(self, existing_steps: 'typing.Sequence[StepBase]', available_data_references: typing.AbstractSet[str]) -> None:
        """
        Checks if a step can be added given existing steps and available
        data references to provide to the step. It also checks if the
        state of a step is suitable to be added at this point.

        Raises an exception if check fails.

        Parameters
        ----------
        existing_steps:
            Steps already in the pipeline.
        available_data_references:
            A set of available data references.
        """

    def set_index(self, index: int) -> None:
        if self.index is not None:
            raise exceptions.InvalidArgumentValueError("Index already set to {index}.".format(index=self.index))

        self.index = index

    @abc.abstractmethod
    def get_free_hyperparams(self) -> typing.Union[typing.Dict, typing.Sequence]:
        """
        Returns step's hyper-parameters which have not been fixed by the pipeline.

        Returns
        -------
        Hyper-parameters configuration for free hyper-parameters, or a list of those.
        """

    @abc.abstractmethod
    def get_all_hyperparams(self) -> typing.Union[typing.Dict, typing.Sequence]:
        """
        Returns step's hyper-parameters.

        Returns
        -------
        Hyper-parameters configuration for all hyper-parameters, or a list of those.
        """

    @abc.abstractmethod
    def get_input_data_references(self) -> typing.AbstractSet[str]:
        pass

    @abc.abstractmethod
    def get_output_data_references(self) -> typing.AbstractSet[str]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_json_structure(cls: typing.Type[S], step_description: typing.Dict, *, resolver: Resolver = None) -> S:
        pass

    @abc.abstractmethod
    def to_json_structure(self) -> typing.Dict:
        pass


SP = typing.TypeVar('SP', bound='PrimitiveStep')


class PrimitiveStep(StepBase):
    """
    Class representing a primitive execution step in pipeline's execution.

    Attributes
    ----------
    primitive_description:
        A description of the primitive specified for this step. Available if ``primitive`` could not be resolved.
    primitive:
        A primitive class associated with this step.
    outputs:
        A list of method names providing outputs for this step.
    hyperparams:
        A map of of fixed hyper-parameters to their values which are set
        as part of a pipeline and should not be tuned during hyper-parameter tuning.
    arguments:
        A map between argument name and its description. Description contains
        a data reference of an output of a prior step (or a pipeline input).
    users:
        Users associated with the primitive.

    Parameters
    ----------
    primitive_description:
        A description of the primitive specified for this step. Allowed only if ``primitive`` is not provided.
    primitive:
        A primitive class associated with this step. If not provided, resolved using ``resolver`` from ``primitive_description``.
    """

    primitive_description: typing.Dict
    primitive: typing.Type[base.PrimitiveBase]
    outputs: typing.List[str]
    hyperparams: typing.Dict[str, typing.Dict]
    arguments: typing.Dict[str, typing.Dict]
    users: typing.List[typing.Dict]

    def __init__(self, primitive_description: typing.Dict = None, *, primitive: typing.Type[base.PrimitiveBase] = None, resolver: typing.Optional[Resolver] = None) -> None:
        super().__init__(resolver=resolver)

        if primitive is None:
            if primitive_description is None:
                raise exceptions.InvalidArgumentValueError("\"primitive_description\" and \"primitive\" arguments are both None.")

            primitive = self.resolver.get_primitive(primitive_description)
        elif primitive_description is not None:
            raise exceptions.InvalidArgumentValueError("\"primitive_description\" and \"primitive\" arguments cannot be both provided.")

        if primitive is None:
            # If still "None" it means resolver returned "None".
            # We just store provided primitive description.
            self.primitive_description = primitive_description
            self.primitive = None
        else:
            self.primitive_description = None
            self.primitive = primitive

        self.outputs: typing.List[str] = []
        self.hyperparams: typing.Dict[str, typing.Dict] = {}
        self.arguments: typing.Dict[str, typing.Dict] = {}
        self.users: typing.List[typing.Dict] = []

    @classmethod
    def get_step_type(cls) -> metadata_base.PipelineStepType:
        return metadata_base.PipelineStepType.PRIMITIVE

    def add_argument(self, name: str, argument_type: typing.Any, data_reference: typing.Union[str, typing.Sequence[str]]) -> None:
        """
        Associate a data reference to an argument of this step (and underlying primitive).

        Parameters
        ----------
        name:
            Argument name.
        argument_type:
            Argument type.
        data_reference:
            Data reference or a list of data references associated with this argument.
        """

        if name in self.arguments:
            raise exceptions.InvalidArgumentValueError("Argument with name '{name}' already exists.".format(name=name))

        if argument_type not in [metadata_base.ArgumentType.CONTAINER, metadata_base.ArgumentType.DATA]:
            raise exceptions.InvalidArgumentValueError("Invalid argument type: {argument_type}".format(argument_type=argument_type))

        if not isinstance(data_reference, str) and not utils.is_instance(data_reference, typing.Sequence[str]):
            raise exceptions.InvalidArgumentTypeError("Data reference is not a string or a list of strings.".format(name=name))

        if self.primitive is not None:
            argument_metadata = self.primitive.metadata.query()['primitive_code'].get('arguments', {}).get(name, None)

            if argument_metadata is None:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown argument name '{name}' for primitive {primitive}.".format(
                        name=name,
                        primitive=self.primitive,
                    ),
                )

            if argument_metadata['kind'] != metadata_base.PrimitiveArgumentKind.PIPELINE:
                raise exceptions.InvalidArgumentValueError(
                    "Pipelines can provide only pipeline arguments, '{name}' is of kind {kind}.".format(
                        name=name,
                        kind=argument_metadata['kind'],
                    ),
                )

        self.arguments[name] = {
            'type': argument_type,
            'data': data_reference,
        }

    def add_output(self, output_id: str) -> None:
        """
        Define an output from this step.

        Underlying primitive can have multiple produce methods but not all have to be
        defined as outputs of the step.

        Parameters
        ----------
        output_id:
            A name of the method producing this output.
        """

        if output_id in self.outputs:
            raise exceptions.InvalidArgumentValueError("Output with ID '{output_id}' already exists.".format(output_id=output_id))

        if self.primitive is not None:
            method_metadata = self.primitive.metadata.query()['primitive_code'].get('instance_methods', {}).get(output_id, None)

            if method_metadata is None:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown output ID '{output_id}' for primitive {primitive}.".format(
                        output_id=output_id,
                        primitive=self.primitive,
                    ),
                )

            if method_metadata['kind'] != metadata_base.PrimitiveMethodKind.PRODUCE:
                raise exceptions.InvalidArgumentValueError(
                    "Primitives can output only from produce methods, '{output_id}' is of kind {kind}.".format(
                        output_id=output_id,
                        kind=method_metadata['kind'],
                    ),
                )

        self.outputs.append(output_id)

    def add_hyperparameter(self, name: str, argument_type: typing.Any, data: typing.Any) -> None:
        """
        Associate a value for a hyper-parameter of this step (and underlying primitive).

        Parameters
        ----------
        name:
            Hyper-parameter name.
        argument_type:
            Argument type.
        data:
            Data reference associated with this hyper-parameter, or list of data references, or value itself.
        """

        if name in self.hyperparams:
            raise exceptions.InvalidArgumentValueError("Hyper-parameter with name '{name}' already exists.".format(name=name))

        if self.primitive is not None:
            hyperparams = self.get_primitive_hyperparams()

            if name not in hyperparams.configuration:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                        name=name,
                        primitive=self.primitive,
                    ),
                )

            if argument_type == metadata_base.ArgumentType.VALUE:
                hyperparams.configuration[name].validate(data)

        if argument_type in [metadata_base.ArgumentType.DATA, metadata_base.ArgumentType.PRIMITIVE]:
            if utils.is_sequence(data):
                if not len(data):
                    raise exceptions.InvalidArgumentValueError("An empty list of hyper-paramater values.")

        self.hyperparams[name] = {
            'type': argument_type,
            'data': data,
        }

    def add_user(self, user_description: typing.Dict) -> None:
        """
        Add a description of user to a list of users associated with the primitive.

        Parameters
        ----------
        user_description:
            User description.
        """

        if 'id' not in user_description:
            raise exceptions.InvalidArgumentValueError("User description is missing user ID.")

        self.users.append(user_description)

    def check_add(self, existing_steps: typing.Sequence[StepBase], available_data_references: typing.AbstractSet[str]) -> None:
        # Order of steps can be arbitrary during execution (given that inputs for a step are available), but we still
        # want some partial order during construction. We want that arguments can already be satisfied by existing steps.
        for argument_description in self.arguments.values():
            if utils.is_sequence(argument_description['data']):
                data_references = argument_description['data']
            else:
                data_references = typing.cast(typing.Sequence, [argument_description['data']])
            for data_reference in data_references:
                if not isinstance(data_reference, str):
                    raise exceptions.InvalidArgumentTypeError("Argument data reference '{data_reference}' is not a string.".format(data_reference=data_reference))
                elif data_reference not in available_data_references:
                    raise exceptions.InvalidPipelineError("Argument data reference '{data_reference}' is not among available data references.".format(
                        data_reference=data_reference,
                    ))

        for hyperparameter_description in self.hyperparams.values():
            if hyperparameter_description['type'] == metadata_base.ArgumentType.DATA:
                if utils.is_sequence(hyperparameter_description['data']):
                    data_references = hyperparameter_description['data']
                else:
                    data_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])
                for data_reference in data_references:
                    if not isinstance(data_reference, str):
                        raise exceptions.InvalidArgumentTypeError("Hyper-parameter data reference '{data_reference}' is not a string.".format(data_reference=data_reference))
                    elif data_reference not in available_data_references:
                        raise exceptions.InvalidPipelineError("Hyper-parameter data reference '{data_reference}' is not among available data references.".format(
                            data_reference=data_reference,
                        ))
            elif hyperparameter_description['type'] == metadata_base.ArgumentType.PRIMITIVE:
                if utils.is_sequence(hyperparameter_description['data']):
                    primitive_references = hyperparameter_description['data']
                else:
                    primitive_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])
                for primitive_reference in primitive_references:
                    if not isinstance(primitive_reference, int):
                        raise exceptions.InvalidArgumentTypeError("Primitive reference '{primitive_reference}' is not an integer.".format(primitive_reference=primitive_reference))
                    elif not 0 <= primitive_reference < len(existing_steps):
                        raise exceptions.InvalidPipelineError("Invalid primitive reference in a step: {primitive}".format(primitive=primitive_reference))
                    elif not isinstance(existing_steps[primitive_reference], PrimitiveStep):
                        raise exceptions.InvalidArgumentTypeError("Primitive reference '{primitive_reference}' is not referencing a primitive step.".format(primitive_reference=primitive_reference))
            elif hyperparameter_description['type'] == metadata_base.ArgumentType.CONTAINER:
                if not isinstance(hyperparameter_description['data'], str):
                    raise exceptions.InvalidArgumentTypeError("Hyper-parameter data reference '{data_reference}' is not a string.".format(
                        data_reference=hyperparameter_description['data'],
                    ))
                elif hyperparameter_description['data'] not in available_data_references:
                    raise exceptions.InvalidPipelineError("Hyper-parameter data reference '{data_reference}' is not among available data references.".format(
                        data_reference=hyperparameter_description['data'],
                    ))
            elif hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE:
                # "VALUE" hyper-parameter value has already been checked in "add_hyperparameter".
                pass
            else:
                raise exceptions.UnexpectedValueError("Unknown hyper-parameter type: {hyperparameter_type}".format(hyperparameter_type=hyperparameter_description['type']))

        # We do this check only if primitive has any arguments or outputs defined.
        # Otherwise it can be used as a unfitted primitive value for a hyper-parameter to another primitive.
        if self.primitive is not None and (self.arguments or self.outputs):
            primitive_arguments = self.primitive.metadata.query()['primitive_code'].get('arguments', {})
            required_arguments_set = {
                argument_name for argument_name, argument in primitive_arguments.items() if 'default' not in argument and argument['kind'] == metadata_base.PrimitiveArgumentKind.PIPELINE
            }

            arguments_set = set(self.arguments.keys())

            missing_arguments_set = required_arguments_set - arguments_set
            if len(missing_arguments_set):
                raise exceptions.InvalidArgumentValueError(
                    "Not all required arguments are provided for the primitive: {missing_arguments_set}".format(
                        missing_arguments_set=missing_arguments_set,
                    )
                )

    def get_primitive_hyperparams(self) -> hyperparams_module.Hyperparams:
        if self.primitive is None:
            raise exceptions.InvalidStateError("Primitive has not been resolved.")

        return self.primitive.metadata.get_hyperparams()

    def get_free_hyperparams(self) -> typing.Dict:
        free_hyperparams = collections.OrderedDict(self.get_primitive_hyperparams().configuration)

        for hyperparam in self.hyperparams:
            del free_hyperparams[hyperparam]

        return free_hyperparams

    def get_all_hyperparams(self) -> typing.Dict:
        return collections.OrderedDict(self.get_primitive_hyperparams().configuration)

    def get_input_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for argument_description in self.arguments.values():
            if utils.is_sequence(argument_description['data']):
                for data_reference in argument_description['data']:
                    data_references.add(data_reference)
            else:
                data_references.add(argument_description['data'])

        for hyperparameter_description in self.hyperparams.values():
            if hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE:
                continue

            if hyperparameter_description['type'] == metadata_base.ArgumentType.PRIMITIVE:
                continue

            if utils.is_sequence(hyperparameter_description['data']):
                for data_reference in hyperparameter_description['data']:
                    data_references.add(data_reference)
            else:
                data_references.add(hyperparameter_description['data'])

        return data_references

    def get_output_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for output_id in self.outputs:
            data_references.add('steps.{i}.{output_id}'.format(i=self.index, output_id=output_id))

        return data_references

    @classmethod
    def from_json_structure(cls: typing.Type[SP], step_description: typing.Dict, *, resolver: typing.Optional[Resolver] = None) -> SP:
        step = cls(step_description['primitive'], resolver=resolver)

        for argument_name, argument_description in step_description.get('arguments', {}).items():
            argument_type = metadata_base.ArgumentType[argument_description['type']]
            step.add_argument(argument_name, argument_type, argument_description['data'])

        for output_description in step_description.get('outputs', []):
            step.add_output(output_description['id'])

        for hyperparameter_name, hyperparameter_description in step_description.get('hyperparams', {}).items():
            argument_type = metadata_base.ArgumentType[hyperparameter_description['type']]

            # If "primitive" is not available, we do not parse the value and we leave it in its JSON form.
            if argument_type == metadata_base.ArgumentType.VALUE and step.primitive is not None:
                hyperparams = step.get_primitive_hyperparams()

                if hyperparameter_name not in hyperparams.configuration:
                    raise exceptions.InvalidArgumentValueError(
                        "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                            name=hyperparameter_name,
                            primitive=step.primitive,
                        ),
                    )

                data = hyperparams.configuration[hyperparameter_name].value_from_json_structure(hyperparameter_description['data'])

            else:
                data = hyperparameter_description['data']

            step.add_hyperparameter(hyperparameter_name, argument_type, data)

        for user_description in step_description.get('users', []):
            step.add_user(user_description)

        return step

    def _output_to_json_structure(self, output_id: str) -> typing.Dict:
        return {'id': output_id}

    def _hyperparameter_to_json_structure(self, hyperparameter_name: str) -> typing.Dict:
        hyperparameter_description = copy.copy(self.hyperparams[hyperparameter_name])

        hyperparameter_description['type'] = hyperparameter_description['type'].name

        # If "primitive" is not available, we have the value already in its JSON form.
        if hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE and self.primitive is not None:
            hyperparams = self.get_primitive_hyperparams()

            if hyperparameter_name not in hyperparams.configuration:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                        name=hyperparameter_name,
                        primitive=self.primitive,
                    ),
                )

            hyperparameter_description['data'] = hyperparams.configuration[hyperparameter_name].value_to_json_structure(hyperparameter_description['data'])

        return hyperparameter_description

    def _argument_to_json_structure(self, argument_name: str) -> typing.Dict:
        argument_description = copy.copy(self.arguments[argument_name])

        argument_description['type'] = argument_description['type'].name

        return argument_description

    def to_json_structure(self) -> typing.Dict:
        if self.primitive is None:
            primitive_description = self.primitive_description
        else:
            primitive_metadata = self.primitive.metadata.query()
            primitive_description = {
                'id': primitive_metadata['id'],
                'version': primitive_metadata['version'],
                'python_path': primitive_metadata['python_path'],
                'name': primitive_metadata['name'],
            }

            if 'digest' in primitive_metadata:
                primitive_description['digest'] = primitive_metadata['digest']

        step_description = {
            'type': self.get_step_type().name,
            'primitive': primitive_description,
        }

        if self.arguments:
            step_description['arguments'] = {argument_name: self._argument_to_json_structure(argument_name) for argument_name in self.arguments.keys()}

        if self.outputs:
            step_description['outputs'] = [self._output_to_json_structure(output_id) for output_id in self.outputs]

        if self.hyperparams:
            hyperparams = {}

            for hyperparameter_name in self.hyperparams.keys():
                hyperparams[hyperparameter_name] = self._hyperparameter_to_json_structure(hyperparameter_name)

            step_description['hyperparams'] = hyperparams

        if self.users:
            step_description['users'] = self.users

        return step_description

    def get_primitive_id(self) -> str:
        if self.primitive is not None:
            return self.primitive.metadata.query()['id']
        else:
            return self.primitive_description['id']


SS = typing.TypeVar('SS', bound='SubpipelineStep')


class SubpipelineStep(StepBase):
    def __init__(self, pipeline_description: typing.Dict = None, *, pipeline: 'Pipeline' = None, resolver: typing.Optional[Resolver] = None) -> None:
        super().__init__(resolver=resolver)

        if pipeline is None:
            if pipeline_description is None:
                raise exceptions.InvalidArgumentValueError("\"pipeline_description\" and \"pipeline\" arguments are both None.")

            pipeline = self.resolver.get_pipeline(pipeline_description)
        elif pipeline_description is not None:
            raise exceptions.InvalidArgumentValueError("\"pipeline_description\" and \"pipeline\" arguments cannot be both provided.")

        if pipeline is None:
            # If still "None" it means resolver returned "None".
            # We just store provided pipeline description.
            self.pipeline_description = pipeline_description
            self.pipeline = None
        else:
            self.pipeline_description = None
            self.pipeline = pipeline

        self.inputs: typing.List[str] = []
        self.outputs: typing.List[typing.Optional[str]] = []

    @classmethod
    def get_step_type(cls) -> metadata_base.PipelineStepType:
        return metadata_base.PipelineStepType.SUBPIPELINE

    def add_input(self, data_reference: str) -> None:
        if self.pipeline is not None:
            if len(self.inputs) == len(self.pipeline.inputs):
                raise exceptions.InvalidArgumentValueError("All pipeline's inputs are already provided.")

        self.inputs.append(data_reference)

    def add_output(self, output_id: typing.Optional[str]) -> None:
        """
        Define an output from this step.

        Underlying pipeline can have multiple outputs but not all have to be
        defined as outputs of the step. They can be skipped using ``None``.

        Parameters
        ----------
        output_id:
            ID to be used in the data reference, mapping pipeline's outputs in order.
            If ``None`` this pipeline's output is ignored and not mapped to a data reference.
        """

        if output_id is not None:
            if output_id in self.outputs:
                raise exceptions.InvalidArgumentValueError("Output with ID '{output_id}' already exists.".format(output_id=output_id))

        if self.pipeline is not None:
            if len(self.outputs) == len(self.pipeline.outputs):
                raise exceptions.InvalidArgumentValueError("All pipeline's outputs are already mapped.")

        self.outputs.append(output_id)

    def check_add(self, existing_steps: 'typing.Sequence[StepBase]', available_data_references: typing.AbstractSet[str]) -> None:
        # Order of steps can be arbitrary during execution (given that inputs for a step are available), but we still
        # want some partial order during construction. We want that arguments can already be satisfied by existing steps.
        for data_reference in self.inputs:
            if not isinstance(data_reference, str):
                raise exceptions.InvalidArgumentTypeError("Input data reference '{data_reference}' is not a string.".format(data_reference=data_reference))
            elif data_reference not in available_data_references:
                raise exceptions.InvalidPipelineError("Input data reference '{data_reference}' is not among available data references.".format(data_reference=data_reference))

        # TODO: Check that all inputs are satisfied?

    def get_free_hyperparams(self) -> typing.Sequence:
        if self.pipeline is None:
            raise exceptions.InvalidStateError("Pipeline has not been resolved.")

        return self.pipeline.get_free_hyperparams()

    def get_all_hyperparams(self) -> typing.Sequence:
        if self.pipeline is None:
            raise exceptions.InvalidStateError("Pipeline has not been resolved.")

        return self.pipeline.get_all_hyperparams()

    def get_input_data_references(self) -> typing.AbstractSet[str]:
        return set(self.inputs)

    def get_output_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for output_id in self.outputs:
            if output_id is not None:
                data_references.add('steps.{i}.{output_id}'.format(i=self.index, output_id=output_id))

        return data_references

    @classmethod
    def from_json_structure(cls: typing.Type[SS], step_description: typing.Dict, *, resolver: Resolver = None) -> SS:
        step = cls(step_description['pipeline'], resolver=resolver)

        for input_description in step_description['inputs']:
            step.add_input(input_description['data'])

        for output_description in step_description['outputs']:
            step.add_output(output_description.get('id', None))

        return step

    def _input_to_json_structure(self, data_reference: str) -> typing.Dict:
        return {'data': data_reference}

    def _output_to_json_structure(self, output_id: typing.Optional[str]) -> typing.Dict:
        if output_id is None:
            return {}
        else:
            return {'id': output_id}

    def to_json_structure(self, *, nest_subpipelines: bool = False) -> typing.Dict:
        if nest_subpipelines:
            if self.pipeline is None:
                raise exceptions.InvalidStateError("Pipeline has not been resolved.")

            pipeline_description = self.pipeline._to_json_structure(nest_subpipelines=True)
        elif self.pipeline is None:
            pipeline_description = self.pipeline_description
        else:
            pipeline_description = {
                'id': self.pipeline.id,
                'digest': self.pipeline.get_digest(),
            }

        step_description = {
            'type': self.get_step_type().name,
            'pipeline': pipeline_description,
            'inputs': [self._input_to_json_structure(data_reference) for data_reference in self.inputs],
            'outputs': [self._output_to_json_structure(output_id) for output_id in self.outputs],
        }

        return step_description

    def get_pipeline_id(self) -> str:
        if self.pipeline is not None:
            return self.pipeline.id
        else:
            return self.pipeline_description['id']


SL = typing.TypeVar('SL', bound='PlaceholderStep')


class PlaceholderStep(StepBase):
    def __init__(self, resolver: Resolver = None) -> None:
        super().__init__(resolver=resolver)

        self.inputs: typing.List[str] = []
        self.outputs: typing.List[str] = []

    @classmethod
    def get_step_type(cls) -> metadata_base.PipelineStepType:
        return metadata_base.PipelineStepType.PLACEHOLDER

    def add_input(self, data_reference: str) -> None:
        self.inputs.append(data_reference)

    def add_output(self, output_id: str) -> None:
        if output_id in self.outputs:
            raise exceptions.InvalidArgumentValueError("Output with ID '{output_id}' already exists.".format(output_id=output_id))

        self.outputs.append(output_id)

    def check_add(self, existing_steps: 'typing.Sequence[StepBase]', available_data_references: typing.AbstractSet[str]) -> None:
        # Order of steps can be arbitrary during execution (given that inputs for a step are available), but we still
        # want some partial order during construction. We want that arguments can already be satisfied by existing steps.
        for data_reference in self.inputs:
            if not isinstance(data_reference, str):
                raise exceptions.InvalidArgumentTypeError("Input data reference '{data_reference}' is not a string.".format(data_reference=data_reference))
            elif data_reference not in available_data_references:
                raise exceptions.InvalidArgumentValueError("Input data reference '{data_reference}' is not among available data references.".format(data_reference=data_reference))

    def get_free_hyperparams(self) -> typing.Sequence:
        return []

    def get_all_hyperparams(self) -> typing.Sequence:
        return []

    def get_input_data_references(self) -> typing.AbstractSet[str]:
        return set(self.inputs)

    def get_output_data_references(self) -> typing.AbstractSet[str]:
        data_references = set()

        for output_id in self.outputs:
            data_references.add('steps.{i}.{output_id}'.format(i=self.index, output_id=output_id))

        return data_references

    @classmethod
    def from_json_structure(cls: typing.Type[SL], step_description: typing.Dict, *, resolver: Resolver = None) -> SL:
        step = cls(resolver=resolver)

        for input_description in step_description['inputs']:
            step.add_input(input_description['data'])

        for output_description in step_description['outputs']:
            step.add_output(output_description['id'])

        return step

    def _input_to_json_structure(self, data_reference: str) -> typing.Dict:
        return {'data': data_reference}

    def _output_to_json_structure(self, output_id: str) -> typing.Dict:
        return {'id': output_id}

    def to_json_structure(self) -> typing.Dict:
        step_description = {
            'type': self.get_step_type().name,
            'inputs': [self._input_to_json_structure(data_reference) for data_reference in self.inputs],
            'outputs': [self._output_to_json_structure(output_id) for output_id in self.outputs],
        }

        return step_description


P = typing.TypeVar('P', bound='Pipeline')


class Pipeline:
    """
    Class representing a pipeline.

    Attributes
    ----------
    id:
        A unique ID to identify this pipeline.
    created:
        Timestamp of pipeline creation in UTC timezone.
    source:
        Description of source.
    name:
        Name of the pipeline.
    description:
        Description of the pipeline.
    users:
        Users associated with the pipeline.
    inputs:
        A sequence of input descriptions which provide names for pipeline inputs.
    outputs:
        A sequence of output descriptions which provide data references for pipeline outputs.
    steps:
        A sequence of steps defining this pipeline.

    Parameters
    ----------
    pipeline_id:
        Optional ID for the pipeline. If not provided, it is automatically generated.
    context:
        DEPRECATED: argument ignored.
    created:
        Optional timestamp of pipeline creation in UTC timezone. If not provided, the current time will be used.
    source:
        Description of source. Optional.
    name:
        Name of the pipeline. Optional.
    description:
        Description of the pipeline. Optional.
    """

    id: str
    created: datetime.datetime
    source: typing.Dict
    name: str
    description: str
    users: typing.List[typing.Dict]
    inputs: typing.List[typing.Dict]
    outputs: typing.List[typing.Dict]
    steps: typing.List[StepBase]

    @deprecate.arguments('context', message="argument ignored")
    def __init__(
        self, pipeline_id: str = None, *, context: metadata_base.Context = None,
        created: datetime.datetime = None, source: typing.Dict = None, name: str = None,
        description: str = None
    ) -> None:
        if pipeline_id is None:
            pipeline_id = str(uuid.uuid4())

        if created is None:
            created = datetime.datetime.now(datetime.timezone.utc)
        elif created.tzinfo is None or created.tzinfo.utcoffset(created) is None:
            raise exceptions.InvalidArgumentValueError("'created' timestamp is missing timezone information.")
        else:
            # Convert to UTC timezone and set "tzinfo" to "datetime.timezone.utc".
            created = created.astimezone(datetime.timezone.utc)

        self.id = pipeline_id
        self.created = created
        self.source = source
        self.name = name
        self.description = description

        self.inputs: typing.List[typing.Dict] = []
        self.outputs: typing.List[typing.Dict] = []
        self.steps: typing.List[StepBase] = []
        self.users: typing.List[typing.Dict] = []

    def add_input(self, name: str = None) -> str:
        """
        Add an input to the pipeline.

        Parameters
        ----------
        name:
            Optional human friendly name for the input.

        Returns
        -------
        Data reference for the input added.
        """

        input_description = {}

        if name is not None:
            input_description['name'] = name

        self.inputs.append(input_description)

        return 'inputs.{i}'.format(i=len(self.inputs) - 1)

    def add_output(self, data_reference: str, name: str = None) -> str:
        """
        Add an output to the pipeline.

        Parameters
        ----------
        data_reference:
            Data reference to use as an output.
        name:
            Optional human friendly name for the output.

        Returns
        -------
        Data reference for the output added.
        """

        if data_reference not in self.get_available_data_references():
            raise exceptions.InvalidArgumentValueError("Invalid data reference '{data_reference}'.".format(data_reference=data_reference))

        output_description = {
            'data': data_reference,
        }

        if name is not None:
            output_description['name'] = name

        self.outputs.append(output_description)

        return 'outputs.{i}'.format(i=len(self.outputs) - 1)

    def add_step(self, step: StepBase) -> None:
        """
        Add a step to the sequence of steps in the pipeline.

        Parameters
        ----------
        step:
            A step to add.
        """

        if not isinstance(step, StepBase):
            raise exceptions.InvalidArgumentTypeError("Step is not an instance of StepBase.")

        step.set_index(len(self.steps))

        try:
            step.check_add(self.steps, self.get_available_data_references())
        except Exception as error:
            raise exceptions.InvalidArgumentValueError("Cannot add step {step_index}.".format(step_index=step.index)) from error

        self.steps.append(step)

    def replace_step(self, index: int, replacement_step: StepBase) -> None:
        """
        Replace an existing step (generally a placeholder) with a new step
        (generally a subpipeline). It makes sure that all inputs are available
        at that point in the pipeline, and all outputs needed later from this
        step stay available after replacement.

        If the old pipeline (one before the step being replaced) has already been
        made public under some ID, make sure that new pipeline (one with replaced
        step) has a new different ID before making it public.

        Parameters
        ----------
        index:
            Index of the step to replace.
        replacement_step:
            A new step.
        """

        # TODO: Handle the case when there is a primitive reference to this step (which is a primitive step in such case).
        #       If we are replacing it with a sub-pipeline or placeholder, we should fail.

        if not 0 <= index < len(self.steps):
            raise exceptions.InvalidArgumentValueError("Step index does not point to an existing step.")

        if not isinstance(replacement_step, StepBase):
            raise exceptions.InvalidArgumentTypeError("Step is not an instance of StepBase.")

        replacement_step.set_index(index)

        try:
            replacement_step.check_add(self.steps[0:index], self.get_available_data_references(index))
        except Exception as error:
            raise exceptions.InvalidArgumentValueError("Cannot replace step {step_index}.".format(step_index=index)) from error

        # Which inputs are needed later on?
        later_input_data_references: typing.Set[str] = set()
        for step in self.steps[index + 1:]:
            later_input_data_references.update(step.get_input_data_references())

        # Compute which data references needed later are contributed by existing step?
        used_output_data_references = self.steps[index].get_output_data_references() & later_input_data_references

        # A replacement step has to contribute at least those data references as well.
        if not replacement_step.get_output_data_references() >= used_output_data_references:
            raise exceptions.InvalidArgumentValueError("Cannot replace step {step_index}. Replacement step is not providing needed outputs: {missing_outputs}".format(
                step_index=index,
                missing_outputs=sorted(used_output_data_references - replacement_step.get_output_data_references()),
            ))

        self.steps[index] = replacement_step

    def add_user(self, user_description: typing.Dict) -> None:
        """
        Add a description of user to a list of users associated with the pipeline.

        Parameters
        ----------
        user_description:
            User description.
        """

        if 'id' not in user_description:
            raise exceptions.InvalidArgumentValueError("User description is missing user ID.")

        self.users.append(user_description)

    def get_free_hyperparams(self) -> typing.Sequence:
        """
        Returns pipeline's hyper-parameters which have not been fixed by the pipeline as
        a list of free hyper-parameters for each step, in order of steps.

        Returns
        -------
        A list of hyper-parameters configuration for free hyper-parameters for each step.
        """

        return [step.get_free_hyperparams() for step in self.steps]

    def get_all_hyperparams(self) -> typing.Sequence:
        """
        Returns pipeline's hyper-parameters as a list of hyper-parameters
        for each step, in order of steps.

        Returns
        -------
        A list of hyper-parameters configuration for all hyper-parameters for each step.
        """

        return [step.get_all_hyperparams() for step in self.steps]

    def has_placeholder(self) -> bool:
        """
        Returns ``True`` if the pipeline has a placeholder step, in the pipeline itself, or any subpipeline.

        Returns
        -------
        ``True`` if the pipeline has a placeholder step.
        """

        for step in self.steps:
            if isinstance(step, PlaceholderStep):
                return True
            elif isinstance(step, SubpipelineStep):
                if step.pipeline is None:
                    raise exceptions.InvalidStateError("Pipeline has not been resolved.")
                elif step.pipeline.has_placeholder():
                    return True

        return False

    def get_available_data_references(self, for_step: int = None) -> typing.AbstractSet[str]:
        """
        Returns a set of data references provided by existing steps (and pipeline inputs).

        Those data references can be used by consequent steps as their inputs.

        Attributes
        ----------
        for_step:
            Instead of using all existing steps, use only steps until ``for_step`` step.

        Returns
        -------
        A set of data references.
        """

        data_references = set()

        for i, input_description in enumerate(self.inputs):
            data_references.add('inputs.{i}'.format(i=i))

        for step in self.steps[0:for_step]:
            output_data_references = step.get_output_data_references()

            existing_data_references = data_references & output_data_references
            if existing_data_references:
                raise exceptions.InvalidPipelineError("Steps have overlapping output data references: {existing_data_references}".format(existing_data_references=existing_data_references))

            data_references.update(output_data_references)

        return data_references

    @deprecate.function(message="use get_producing_outputs method instead")
    def get_exposable_outputs(self) -> typing.AbstractSet[str]:
        return self.get_producing_outputs()

    def get_producing_outputs(self) -> typing.AbstractSet[str]:
        """
        Returns a set of recursive data references of all values produced by the pipeline
        during its run.

        This represents outputs of each step of the pipeline, the outputs of the pipeline
        itself, but also exposable outputs of any sub-pipeline. The latter are prefixed with
        the step prefix, e.g., ``steps.1.steps.4.produce`` is ``steps.4.produce`` output
        of a sub-pipeline step with index 1.

        Outputs of sub-pipelines are represented twice, as an output of the step and
        as an output of the sub-pipeline. This is done because not all outputs of a sub-pipeline
        are necessary exposed as an output of a step because they might not be used in
        the outer pipeline, but the sub-pipeline still defines them.

        A primitive might have additional produce methods which could be called but they
        are not listed among step's outputs. Data references related to those produce
        methods are not returned.

        Returns
        -------
        A set of recursive data references.
        """

        exposable_outputs: typing.Set[str] = set()

        for step_index, step in enumerate(self.steps):
            output_data_references = set(step.get_output_data_references())

            if isinstance(step, SubpipelineStep):
                for exposable_output in step.pipeline.get_producing_outputs():
                    output_data_references.add('steps.{step_index}.{exposable_output}'.format(
                        step_index=step_index,
                        exposable_output=exposable_output,
                    ))

            existing_data_references = exposable_outputs & output_data_references
            if existing_data_references:
                raise exceptions.InvalidPipelineError("Steps have overlapping exposable data references: {existing_data_references}".format(existing_data_references=existing_data_references))

            exposable_outputs.update(output_data_references)

        for i, output_description in enumerate(self.outputs):
            exposable_outputs.add('outputs.{i}'.format(i=i))

        return exposable_outputs

    def check(self, *, allow_placeholders: bool = False, standard_pipeline: bool = True, input_types: typing.Dict[str, type] = None) -> None:
        """
        Check if the pipeline is a valid pipeline.

        It supports checking against non-resolved primitives and pipelines, but in that case
        checking will be very limited. Make sure you used a strict resolver to assure
        full checking of this pipeline and any sub-pipelines.

        Raises an exception if check fails.

        Parameters
        ----------
        allow_placeholders:
            Do we allow placeholders in a pipeline?
        standard_pipeline:
            Check it as a standard pipeline (inputs are Dataset objects, output is a DataFrame)?
        input_types:
            A map of types available as inputs. If provided, overrides ``standard_pipeline``.
        """

        self._check(allow_placeholders, standard_pipeline, input_types)

    def _check(self, allow_placeholders: bool, standard_pipeline: bool, input_types: typing.Optional[typing.Dict[str, type]]) -> typing.Sequence[TypeInfo]:
        # Generating JSON also checks it against the pipeline schema.
        # We do not set "nest_subpipelines" because recursive checks are done
        # by this method's recursive call (when sub-pipelines are resolved).
        self.to_json_structure()

        # Map between available data references and their types.
        environment: typing.Dict[str, TypeInfo] = {}

        # Inputs are never singleton.
        if input_types is not None:
            if len(self.inputs) != len(input_types):
                raise exceptions.InvalidPipelineError("Pipeline '{pipeline_id}' accepts {inputs} input(s), but {input_types} provided.".format(
                    pipeline_id=self.id,
                    inputs=len(self.inputs),
                    input_types=len(input_types),
                ))

            for data_reference, structural_type in input_types.items():
                environment[data_reference] = TypeInfo(structural_type, False)
        elif standard_pipeline:
            for i, input_description in enumerate(self.inputs):
                environment['inputs.{i}'.format(i=i)] = TypeInfo(container.Dataset, False)
        else:
            for i, input_description in enumerate(self.inputs):
                # We do not really know what the inputs are.
                environment['inputs.{i}'.format(i=i)] = TypeInfo(typing.Any, False)  # type: ignore

        for step_index, step in enumerate(self.steps):
            assert step_index == step.index

            if isinstance(step, PlaceholderStep):
                if not allow_placeholders:
                    raise exceptions.InvalidPipelineError("Step {step_index} of pipeline '{pipeline_id}' is a placeholder but there should be no placeholders.".format(
                        step_index=step_index,
                        pipeline_id=self.id,
                    ))

                for data_reference in step.inputs:
                    # This is checked already during pipeline construction in "check_add".
                    assert data_reference in environment

                for data_reference in step.get_output_data_references():
                    # This is checked already during pipeline construction in "add_output".
                    assert data_reference not in environment

                    # We cannot really know a type of the placeholder output given current pipeline description.
                    environment[data_reference] = TypeInfo(typing.Any, None)  # type: ignore

            elif isinstance(step, SubpipelineStep):
                subpipeline_input_types: typing.Dict[str, type] = {}
                for i, data_reference in enumerate(step.inputs):
                    # This is checked already during pipeline construction in "check_add".
                    assert data_reference in environment

                    input_data_reference = 'inputs.{i}'.format(i=i)

                    assert input_data_reference not in subpipeline_input_types
                    subpipeline_input_types[input_data_reference] = environment[data_reference].structural_type

                # Resolving is optional. Of course full checking is not really possible without resolving.
                if step.pipeline is not None:
                    outputs_types = step.pipeline._check(allow_placeholders, False, subpipeline_input_types)

                for i, output_id in enumerate(step.outputs):
                    if output_id is not None:
                        output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

                        # This is checked already during pipeline construction in "add_output".
                        assert output_data_reference not in environment

                        if step.pipeline is not None:
                            environment[output_data_reference] = outputs_types[i]
                        else:
                            # We cannot really know a type of the output without resolving.
                            environment[output_data_reference] = TypeInfo(typing.Any, None)  # type: ignore

            elif isinstance(step, PrimitiveStep):
                if step.primitive is not None:
                    primitive_metadata = step.primitive.metadata.query()
                    primitive_methods = primitive_metadata['primitive_code'].get('instance_methods', {})
                    primitive_arguments = primitive_metadata['primitive_code'].get('arguments', {})

                for argument_name, argument_description in step.arguments.items():
                    # This is checked already during pipeline construction in "check_add".
                    if utils.is_sequence(argument_description['data']):
                        for data_reference in argument_description['data']:
                            assert data_reference in environment
                    else:
                        assert argument_description['data'] in environment

                    if step.primitive is not None:
                        # This is checked already during pipeline construction in "add_argument".
                        assert argument_name in primitive_arguments

                    if argument_description['type'] == metadata_base.ArgumentType.DATA:
                        type_info = environment[argument_description['data']]

                        # The error is only if it is exactly "False". If it is "None", we do not know and we do not want any false positives.
                        if type_info.singleton == False:  # noqa
                            raise exceptions.InvalidPipelineError(
                                "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data reference is not.".format(
                                    argument_name=argument_name,
                                    step_index=step_index,
                                    pipeline_id=self.id,
                                ),
                            )

                        # We cannot really check if types match because we do not know
                        # the type of elements from just container structural type.
                    elif step.primitive is not None:
                        assert argument_description['type'] == metadata_base.ArgumentType.CONTAINER, argument_description['type']

                        if utils.is_sequence(argument_description['data']):
                            if not utils.is_subclass(primitive_arguments[argument_name]['type'], container.List):
                                raise exceptions.InvalidPipelineError(
                                    "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' should have type 'List' to support getting a list of values, "
                                    "but it has type '{argument_type}'.".format(
                                        argument_name=argument_name,
                                        step_index=step_index,
                                        pipeline_id=self.id,
                                        argument_type=primitive_arguments[argument_name]['type'],
                                    ),
                                )

                        else:
                            type_info = environment[argument_description['data']]

                            if type_info.structural_type is typing.Any or primitive_arguments[argument_name]['type'] is typing.Any:
                                # No type information.
                                pass
                            elif not utils.is_subclass(type_info.structural_type, primitive_arguments[argument_name]['type']):
                                raise exceptions.InvalidPipelineError(
                                    "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' has type '{argument_type}', but it is getting a type '{input_type}'.".format(
                                        argument_name=argument_name,
                                        step_index=step_index,
                                        pipeline_id=self.id,
                                        argument_type=primitive_arguments[argument_name]['type'],
                                        input_type=type_info.structural_type,
                                    ),
                                )

                if step.primitive is not None:
                    hyperparams = step.get_primitive_hyperparams()

                    for hyperparameter_name, hyperparameter_description in step.hyperparams.items():
                        # This is checked already during pipeline construction in "add_hyperparameter".
                        assert hyperparameter_name in hyperparams.configuration

                        if hyperparameter_description['type'] == metadata_base.ArgumentType.DATA:
                            if utils.is_sequence(hyperparameter_description['data']):
                                data_references = hyperparameter_description['data']
                            else:
                                data_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])

                            for data_reference in data_references:
                                # This is checked already during pipeline construction in "check_add".
                                assert data_reference in environment

                                if not isinstance(data_reference, str):
                                    raise exceptions.InvalidArgumentTypeError("Hyper-parameter data reference '{data_reference}' is not a string.".format(data_reference=data_reference))

                                type_info = environment[data_reference]

                                # The error is only if it is exactly "False". If it is "None", we do not know and we do not want any false positives.
                                if type_info.singleton == False:  # noqa
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, "
                                        "but available data reference '{data_reference}' is not.".format(
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            data_reference=data_reference,
                                        ),
                                    )

                                # We cannot really check if types match because we do not know
                                # the type of elements from just container structural type.

                        elif hyperparameter_description['type'] == metadata_base.ArgumentType.PRIMITIVE:
                            if utils.is_sequence(hyperparameter_description['data']):
                                primitive_references = hyperparameter_description['data']
                            else:
                                primitive_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])

                            primitives = []
                            for primitive_reference in primitive_references:
                                # This is checked already during pipeline construction in "check_add".
                                assert 0 <= primitive_reference < step_index

                                primitive_step = self.steps[primitive_reference]

                                if not isinstance(primitive_step, PrimitiveStep):
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' "
                                        "does not point to a primitive step (step {primitive_reference}).".format(
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            primitive_reference=primitive_reference,
                                        ),
                                    )

                                if primitive_step.primitive is None:
                                    primitives.append(typing.Any)
                                else:
                                    primitives.append(primitive_step.primitive)

                            if utils.is_sequence(hyperparameter_description['data']):
                                if not hyperparams.configuration[hyperparameter_name].can_accept_value_type(primitives):
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' cannot accept primitives {primitives}.".format(
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            primitives=primitives,
                                        ),
                                    )
                            else:
                                assert len(primitives) == 1

                                if not hyperparams.configuration[hyperparameter_name].can_accept_value_type(primitives[0]):
                                    raise exceptions.InvalidPipelineError(
                                        "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' cannot accept a primitive '{primitive}'.".format(
                                            hyperparameter_name=hyperparameter_name,
                                            step_index=step_index,
                                            pipeline_id=self.id,
                                            primitive=primitives[0],
                                        ),
                                    )

                        elif hyperparameter_description['type'] == metadata_base.ArgumentType.CONTAINER:
                            # This is checked already during pipeline construction in "check_add".
                            assert hyperparameter_description['data'] in environment

                            type_info = environment[hyperparameter_description['data']]

                            if not hyperparams.configuration[hyperparameter_name].can_accept_value_type(type_info.structural_type):
                                raise exceptions.InvalidPipelineError(
                                    "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' cannot accept a value of type '{input_type}'.".format(
                                        hyperparameter_name=hyperparameter_name,
                                        step_index=step_index,
                                        pipeline_id=self.id,
                                        input_type=type_info.structural_type,
                                    ),
                                )

                        elif hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE:
                            # "VALUE" hyper-parameter value has already been checked in "add_hyperparameter".
                            pass

                        else:
                            raise exceptions.UnexpectedValueError("Unknown hyper-parameter type: {hyperparameter_type}".format(hyperparameter_type=hyperparameter_description['type']))

                for output_id in step.outputs:
                    output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

                    assert output_data_reference not in environment

                    if step.primitive is not None:
                        # This is checked already during pipeline construction in "add_output".
                        assert output_id in primitive_methods

                        method_description = primitive_methods[output_id]

                        produce_type = method_description['returns']

                        # This should be checked by some other part of the code (like primitive validation).
                        assert issubclass(produce_type, base.CallResult), produce_type

                        output_type = utils.get_type_arguments(produce_type)[base.T]  # type: ignore

                        environment[output_data_reference] = TypeInfo(output_type, method_description.get('singleton', False))
                    else:
                        # We cannot really know a type of the output without resolving.
                        environment[output_data_reference] = TypeInfo(typing.Any, None)  # type: ignore

            else:
                raise exceptions.UnexpectedValueError("Unknown step type: {step_type}".format(step_type=type(step)))

        outputs_types = []
        for output_description in self.outputs:
            # This is checked already during pipeline construction in "add_output".
            assert output_description['data'] in environment, output_description['data']

            outputs_types.append(environment[output_description['data']])

        return outputs_types

    @classmethod
    def from_yaml(cls: typing.Type[P], string_or_file: typing.Union[str, typing.IO[typing.Any]], *, resolver: typing.Optional[Resolver] = None,
                  strict_digest: bool = False) -> P:
        description = utils.yaml_load(string_or_file)

        return cls.from_json_structure(description, resolver=resolver, strict_digest=strict_digest)

    @classmethod
    def from_json(cls: typing.Type[P], string_or_file: typing.Union[str, typing.IO[typing.Any]], *, resolver: typing.Optional[Resolver] = None,
                  strict_digest: bool = False) -> P:
        if isinstance(string_or_file, str):
            description = json.loads(string_or_file)
        else:
            description = json.load(string_or_file)

        return cls.from_json_structure(description, resolver=resolver, strict_digest=strict_digest)

    @classmethod
    def _get_step_class(cls, step_type: typing.Any) -> StepBase:
        if step_type == metadata_base.PipelineStepType.PRIMITIVE:
            return PrimitiveStep
        elif step_type == metadata_base.PipelineStepType.SUBPIPELINE:
            return SubpipelineStep
        elif step_type == metadata_base.PipelineStepType.PLACEHOLDER:
            return PlaceholderStep
        else:
            raise exceptions.InvalidArgumentValueError("Invalid step type '{step_type}'.".format(step_type=step_type))

    @classmethod
    def _get_source(cls, pipeline_description: typing.Dict) -> typing.Optional[typing.Dict]:
        return pipeline_description.get('source', None)

    @classmethod
    def _canonical_pipeline_description(cls, pipeline_description: typing.Dict) -> typing.Dict:
        """
        Before we compute digest of the pipeline description, we have to convert it to a
        canonical structure.

        Currently, this is just removing any sub-pipelines the description might have nested.
        """

        pipeline_description = copy.deepcopy(pipeline_description)

        for step_description in pipeline_description['steps']:
            if step_description['type'] == metadata_base.PipelineStepType.SUBPIPELINE:
                new_description = {
                    'id': step_description['pipeline']['id'],
                }
                if 'digest' in step_description['pipeline']:
                    new_description['digest'] = step_description['pipeline']['digest']
                step_description['pipeline'] = new_description

        # Not really part of pipeline schema, but used in evaluation. Digest should
        # not be computed using it, if it was passed in. We also do not want to store
        # it in metalearning database as part of the pipeline document so that we are
        # not storing same pipeline multiple times, just with different rank values.
        if 'pipeline_rank' in pipeline_description:
            del pipeline_description['pipeline_rank']

        return pipeline_description

    @classmethod
    def from_json_structure(cls: typing.Type[P], pipeline_description: typing.Dict, *, resolver: typing.Optional[Resolver] = None,
                            strict_digest: bool = False) -> P:
        PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_description)

        if 'digest' in pipeline_description:
            digest = utils.compute_digest(cls._canonical_pipeline_description(pipeline_description))

            if digest != pipeline_description['digest']:
                if strict_digest:
                    raise exceptions.DigestMismatchError(
                        "Digest for pipeline '{pipeline_id}' does not match a computed one. Provided digest: {pipeline_digest}. Computed digest: {new_pipeline_digest}.".format(
                            pipeline_id=pipeline_description['id'],
                            pipeline_digest=pipeline_description['digest'],
                            new_pipeline_digest=digest,
                        )
                    )
                else:
                    logger.warning(
                        "Digest for pipeline '%(pipeline_id)s' does not match a computed one. Provided digest: %(pipeline_digest)s. Computed digest: %(new_pipeline_digest)s.",
                        {
                            'pipeline_id': pipeline_description['id'],
                            'pipeline_digest': pipeline_description['digest'],
                            'new_pipeline_digest': digest,
                        },
                    )

        # If no timezone information is provided, we assume UTC. If there is timezone information,
        # we convert timestamp to UTC in the constructor of "Pipeline".
        created = dateparser.parse(pipeline_description['created'], settings={'TIMEZONE': 'UTC'})
        source = cls._get_source(pipeline_description)

        pipeline = cls(
            pipeline_id=pipeline_description['id'], created=created, source=source,
            name=pipeline_description.get('name', None), description=pipeline_description.get('description', None)
        )

        for input_description in pipeline_description['inputs']:
            pipeline.add_input(input_description.get('name', None))

        for step_description in pipeline_description['steps']:
            step = cls._get_step_class(step_description['type']).from_json_structure(step_description, resolver=resolver)
            pipeline.add_step(step)

        for output_description in pipeline_description['outputs']:
            pipeline.add_output(output_description['data'], output_description.get('name', None))

        for user_description in pipeline_description.get('users', []):
            pipeline.add_user(user_description)

        return pipeline

    def _inputs_to_json_structure(self) -> typing.Sequence[typing.Dict]:
        return self.inputs

    def _outputs_to_json_structure(self) -> typing.Sequence[typing.Dict]:
        return self.outputs

    def _source_to_json_structure(self) -> typing.Optional[typing.Dict]:
        return self.source

    def _users_to_json_structure(self) -> typing.Optional[typing.Sequence[typing.Dict]]:
        # Returns "None" if an empty list.
        return self.users or None

    def _to_json_structure(self, *, nest_subpipelines: bool = False) -> typing.Dict:
        # Timestamp should already be in UTC and in particular "tzinfo" should be "datetime.timezone.utc".
        assert self.created.tzinfo == datetime.timezone.utc, self.created
        # We remove timezone information before formatting to not have "+00:00" added and
        # we then manually add "Z" instead (which has equivalent meaning).
        created = self.created.replace(tzinfo=None).isoformat('T') + 'Z'

        pipeline_description: typing.Dict = {
            'id': self.id,
            'schema': PIPELINE_SCHEMA_VERSION,
            'created': created,
            'inputs': self._inputs_to_json_structure(),
            'outputs': self._outputs_to_json_structure(),
            'steps': [],
        }

        source = self._source_to_json_structure()
        if source is not None:
            pipeline_description['source'] = source

        users = self._users_to_json_structure()
        if users is not None:
            pipeline_description['users'] = users

        if self.name is not None:
            pipeline_description['name'] = self.name
        if self.description is not None:
            pipeline_description['description'] = self.description

        for step in self.steps:
            if isinstance(step, SubpipelineStep):
                pipeline_description['steps'].append(step.to_json_structure(nest_subpipelines=nest_subpipelines))
            else:
                pipeline_description['steps'].append(step.to_json_structure())

        pipeline_description['digest'] = utils.compute_digest(self._canonical_pipeline_description(pipeline_description))

        return pipeline_description

    def to_json_structure(self, *, nest_subpipelines: bool = False, canonical: bool = False) -> typing.Dict:
        if canonical:
            nest_subpipelines = False

        pipeline_description = self._to_json_structure(nest_subpipelines=nest_subpipelines)

        if canonical:
            pipeline_description = self._canonical_pipeline_description(pipeline_description)

        PIPELINE_SCHEMA_VALIDATOR.validate(pipeline_description)

        return pipeline_description

    def to_json(self, file: typing.IO[typing.Any] = None, *, nest_subpipelines: bool = False, canonical: bool = False, **kwargs: typing.Any) -> typing.Optional[str]:
        obj = self.to_json_structure(nest_subpipelines=nest_subpipelines, canonical=canonical)

        if 'allow_nan' not in kwargs:
            kwargs['allow_nan'] = False

        if file is None:
            return json.dumps(obj, **kwargs)
        else:
            json.dump(obj, file, **kwargs)
            return None

    def to_yaml(self, file: typing.IO[typing.Any] = None, *, nest_subpipelines: bool = False, canonical: bool = False, **kwargs: typing.Any) -> typing.Optional[str]:
        obj = self.to_json_structure(nest_subpipelines=nest_subpipelines, canonical=canonical)

        return utils.yaml_dump(obj, stream=file, **kwargs)

    def equals(self, pipeline: P, *, strict_order: bool = False, only_control_hyperparams: bool = False) -> bool:
        """
        Check if the two pipelines are equal in the sense of isomorphism.

        Parameters
        ----------
        pipeline:
            A pipeline instance.
        strict_order:
            If true, we will treat inputs of `Set` hyperparameters as a list, and the order of primitives are determined by their step indices.
            Otherwise we will try to sort contents of `Set` hyperparameters so the orders of their contents are not important,
            and we will try topological sorting to determine the order of nodes.
        only_control_hyperparams:
            If true, equality checks will not happen for any hyperparameters that are not of the ``ControlParameter`` semantic type, i.e.
            there will be no checks for hyperparameters that are specific to the hyperparameter optimization phase, and not part of the
            logic of the pipeline.

        Notes
        -----
        This algorithm checks if the two pipelines are equal in the sense of isomorphism by solving a graph isomorphism
        problem. The general graph isomorphism problem is known to be neither P nor NP-complete. However,
        our pipelines are DAGs so we could have an algorithm to check its isomorphism in polynomial time.

        The complexity of this algorithm is around :math:`O((V + E)logV)`, where :math:`V` is the number of steps in the
        pipeline and :math:`E` is the number of output references. It tries to assign unique orders to all nodes layer
        by layer greedily followed by a topological sort using DFS. Then we can get a unique, hashable & comparable
        tuple representing the structure of the pipeline. It is also a unique representation of the equivalence class of
        a pipeline in the sense of isomorphism.
        """

        # TODO: We could cache the representation once the pipeline is freezed.
        return \
            PipelineHasher(self, strict_order, only_control_hyperparams).unique_equivalence_class_repr() == \
            PipelineHasher(pipeline, strict_order, only_control_hyperparams).unique_equivalence_class_repr()

    def hash(self, *, strict_order: bool = False, only_control_hyperparams: bool = False) -> int:
        """
        Get the hash value of a pipeline. It simply hashes the unique representation of the equivalence class of
        a pipeline in the sense of isomorphism.

        strict_order:
            If true, we will treat inputs of `Set` hyperparameters as a list, and the order of primitives are determined by their step indices.
            Otherwise we will try to sort contents of `Set` hyperparameters so the orders of their contents are not important,
            and we will try topological sorting to determine the order of nodes.
        only_control_hyperparams:
            If true, equality checks will not happen for any hyperparameters that are not of the ``ControlParameter`` semantic type, i.e.
            there will be no checks for hyperparameters that are specific to the hyperparameter optimization phase, and not part of the
            logic of the pipeline.
        """

        # TODO: We could cache the hash once the pipeline is freezed.
        return hash(PipelineHasher(self, strict_order, only_control_hyperparams))

    def get_digest(self) -> str:
        return self._to_json_structure(nest_subpipelines=False)['digest']


# There are several forms of input indices.
# 1. Named arguments. They are typically strings or tuple-wrapped strings.
# 2. Pipeline outputs. They are integers.
# 3. Value-type & container-type hyperparameters. They are strings.
# 4. Data-type hyperparameters. They are tuples like (name, type) or (name, type, index).
# 5. Primitive-type hyperparameters. They are strings or tuples like (name, index).
InputIndex = typing.Union[int, str, typing.Tuple[str], typing.Tuple[str, str], typing.Tuple[str, int], typing.Tuple[str, str, int]]
OutputIndex = int
Edge = typing.NamedTuple('Edge', [('input_index', InputIndex), ('output_index', OutputIndex)])
PD = typing.TypeVar('PD', bound='PipelineDAG')


class OrderedNode(metaclass=utils.AbstractMetaclass):
    """This class represents a node in a DAG.

    Parameters
    ----------
    name:
        The name of this node.
    topological_order:
        The topological order of this node in the DAG.
    inputs_ref:
        The inputs containing unresolved reference strings or list of indices.

    Attributes
    ----------
    name:
        The name of this node.
    topological_order:
        The topological order of a node in a DAG.
    global_order:
        The global order of a node in a DAG.
    inputs:
        The inputs of the node. They serve as the edges in a DAG.
    children:
        The descendants of this node.
    """

    name: str
    topological_order: int
    global_order: int
    inputs: typing.Dict
    children: typing.Dict

    def __init__(self, name: str, topological_order: int = 0, inputs_ref: typing.Optional[typing.Union[typing.Dict[InputIndex, str], typing.List[str]]] = None) -> None:
        self.name = name
        self.topological_order: int = topological_order

        if inputs_ref is None:
            inputs_ref = collections.OrderedDict()
        elif isinstance(inputs_ref, list):
            inputs_ref = collections.OrderedDict(enumerate(inputs_ref))
        self._inputs_ref = inputs_ref

        self.global_order: typing.Optional[int] = None
        self.inputs: typing.Dict[InputIndex, typing.Tuple['OrderedNode', int]] = collections.OrderedDict()
        self.children: typing.DefaultDict['OrderedNode', typing.Set[InputIndex]] = collections.defaultdict(set)
        self._frozen = False
        self._unique_equivalence_class_repr: typing.Optional[typing.Tuple] = None

    @property
    def inputs_count(self) -> int:
        """
        Returns the count of inputs.
        """
        return len(self._inputs_ref)

    def outputs(self) -> typing.DefaultDict[OutputIndex, typing.Set[typing.Tuple['OrderedNode', InputIndex]]]:
        reverse_dict: typing.DefaultDict[OutputIndex, typing.Set[typing.Tuple[OrderedNode, InputIndex]]] = collections.defaultdict(set)
        for node, input_indices in self.children.items():
            for input_index in input_indices:
                output_index = node.inputs[input_index][1]
                reverse_dict[output_index].add((node, input_index))
        return reverse_dict

    @property
    def frozen(self) -> bool:
        """
        If a node is frozen, its representation can be cached.

        Returns
        -------
        The frozen state of the node.
        """

        return self._frozen

    @frozen.setter
    def frozen(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._frozen = value
        if not value:
            # Force cleanup.
            self._unique_equivalence_class_repr = None

    def add_child(self, node: 'OrderedNode', edge: Edge) -> None:
        """
        Add a child node.

        Parameters
        ----------
        node:
            The child node.
        edge:
            The edge connects parent node and child node.
        """

        self.children[node].add(edge.input_index)
        node.inputs[edge.input_index] = (self, edge.output_index)

    def remove_child(self, child: 'OrderedNode', input_index: typing.Optional[InputIndex]) -> None:
        """
        Remove a child node.

        Parameters
        ----------
        child:
            The child node.
        input_index:
            The related input index of the child node. If it is None, all edges between the child ndoe and the parent node will be removed.
        """

        if input_index is None:
            for input_index in self.children[child]:
                del child.inputs[input_index]
            del self.children[child]
        else:
            edges = self.children[child]
            edges.remove(input_index)
            del child.inputs[input_index]
            if not edges:
                del self.children[child]

    def change_input(self, input_index: InputIndex, new_parent: 'OrderedNode', new_input_index: typing.Optional[InputIndex] = None, new_output_index: typing.Optional[OutputIndex] = None) -> None:
        """
        Change the input of the node.

        Parameters
        ----------
        input_index:
            The input index we want to change.
        new_parent:
            The new parent of the node.
        new_input_index:
            The new input index. If it is None, the original index will be kept.
        new_output_index:
            The new output index. If it is None, the original index will be kept.
        """

        parent, output_index = self.inputs[input_index]
        parent.remove_child(self, input_index)
        if new_output_index is None:
            new_output_index = output_index
        if new_input_index is None:
            new_input_index = input_index
        else:
            del self.inputs[input_index]
        new_parent.add_child(self, Edge(input_index=new_input_index, output_index=new_output_index))

    def join(self, node_with_inputs: 'OrderedNode') -> None:
        """
        Join by the edges of the nodes.

        Two nodes can be joined only if the output indices of node A (`self` here) match the input indices of node B (`node_with_inputs` here).
        The join operation needs two nodes: A and B. Suppose A's children are {A+} and B's parents are {B-}.

        It removes all edges between B and {B-} & between A and {A+}, then creating new edges to connect {B-} and {A+}.

        Parameters
        ----------
        node_with_inputs:
            The node which provides inputs.

        Notes
        -----
        The function is named ``join`` because it is similar to "join" of SQL since they both concatenate items by their common indices.
        """

        outputs = self.outputs()
        # Set & dict size will be changed during iteration. Use a list to fix them.
        for input_index, (parent, parent_output_index) in list(node_with_inputs.inputs.items()):
            assert isinstance(input_index, int)
            for child, child_input in outputs[input_index]:
                child.change_input(child_input, parent, new_output_index=parent_output_index)
            parent.remove_child(node_with_inputs, input_index)

    @abc.abstractmethod
    def reference_name(self) -> int:
        """
        The name to reference itself.
        """

    @abc.abstractmethod
    def output_reference_names(self) -> typing.List[str]:
        """
        The names for other nodes to refer its outputs.
        """

    def resolve_input_references(self, nodes_outputs_reverse_dict: typing.Dict[str, typing.Tuple['OrderedNode', OutputIndex]]) -> None:
        """
        Resolve input references with a lookup dict.
        """

        for input_index, ref in self._inputs_ref.items():
            parent, output_index = nodes_outputs_reverse_dict[ref]
            parent.add_child(self, Edge(input_index=input_index, output_index=output_index))

    def _unique_ordered_inputs(self) -> typing.Tuple:
        input_orders = [(name, parent.global_order, output_index) for name, (parent, output_index) in self.inputs.items()]
        input_orders.sort()
        return tuple(input_orders)

    def unique_equivalence_class_repr(self) -> typing.Tuple:
        """
        Get the unique representation of the equivalence class of the node in the sense of isomorphism.
        """

        if not self.frozen or self._unique_equivalence_class_repr is None:
            repr_tuple = (self.name, self._unique_ordered_inputs(), self.topological_order)
            if self.frozen:
                self._unique_equivalence_class_repr = repr_tuple
            else:
                self._unique_equivalence_class_repr = None
            return repr_tuple

        return self._unique_equivalence_class_repr


class InputsNode(OrderedNode):
    """This class represents the inputs of a pipeline. This node is unique in a pipeline.

    Parameters
    ----------
    pipeline_inputs:
        Inputs of the pipeline. It is a list contains description dicts of inputs. Their order matters.
        They will not be resolved as data reference strings, so we use `pipeline_inputs` as its name instead of `inputs_ref` which will be resolved.
    """
    def __init__(self, pipeline_inputs: typing.List[typing.Dict]) -> None:
        super().__init__('Inputs')

        self.pipeline_inputs = copy.deepcopy(pipeline_inputs)
        self.global_order = 0

    @property
    def inputs_count(self) -> int:
        """
        Return the count of inputs.
        """
        return len(self.pipeline_inputs)

    def reference_name(self) -> int:
        """
        We specify that the input node has index -1.
        """

        return -1

    def output_reference_names(self) -> typing.List[str]:
        """
        The names for other nodes to refer its outputs.
        """

        return ['inputs.{i}'.format(i=i) for i in range(self.inputs_count)]

    def unique_equivalence_class_repr(self) -> typing.Tuple:
        """
        Get the unique representation of the equivalence class of the node in the sense of isomorphism.
        """

        return self.name, self.inputs_count


class OutputsNode(OrderedNode):
    """This class represents the outputs of a pipeline. This node is unique in a pipeline.

    Parameters
    ----------
    pipeline_outputs:
        Outputs of a pipeline. It is a list contains description dicts of outputs. Their order matters.
    """
    def __init__(self, pipeline_outputs: typing.List[typing.Dict]) -> None:
        super().__init__('Outputs', inputs_ref=[v['data'] for v in pipeline_outputs])

        self.outputs_count = len(pipeline_outputs)

    def reference_name(self) -> int:
        """
        We specify that the output node has index -2.
        """

        return -2

    def output_reference_names(self) -> typing.List[str]:
        """
        The names for other nodes to refer its outputs.
        """

        return []


class PrimitiveNode(OrderedNode):
    """
    This class represents a primitive step in a DAG.

    Attributes
    ----------
    index:
        The index of this step in the pipeline.
    primitive_step:
        The PrimitiveStep instance.
    _steps_ref:
        Raw inputs info contains step reference indices.
    steps:
        Steps used by this node as parameters or hyperparameters.
    values:
        Inputs contains simple value.
    strict_order:
        If true, we will treat inputs of `Set` hyperparameters as a list.
        Otherwise we will try to sort their contents so the orders of their contents are not important.
    only_control_hyperparams:
        If true, hyperparameters that are not of the `ControlParameter` semantic type. will not be included
        in the node's representation.
    """

    index: int
    primitive_step: PrimitiveStep
    _steps_ref: typing.Dict
    steps: typing.Dict
    values: typing.Dict
    strict_order: bool
    only_control_hyperparams: bool

    def __init__(self, primitive: PrimitiveStep, *, strict_order: bool, only_control_hyperparams: bool) -> None:
        # We wraps argument names with a tuple to unify sorting.
        super().__init__(primitive.get_primitive_id(), inputs_ref={(k,): v['data'] for k, v in primitive.arguments.items()})

        self.index: int = primitive.index
        self.primitive_step = primitive
        self.strict_order = strict_order
        self.only_control_hyperparams = only_control_hyperparams

        self._outputs: typing.List[str] = primitive.outputs.copy()
        self._steps_ref: typing.Dict[InputIndex, int] = collections.OrderedDict()
        self.steps: typing.Dict[InputIndex, OrderedNode] = collections.OrderedDict()
        self.values: typing.Dict[str, typing.Any] = collections.OrderedDict()

        if self.primitive_step.primitive is not None:
            hyperparameters = self.primitive_step.get_primitive_hyperparams().configuration
        else:
            hyperparameters = None

        # Resolve hyper-parameters. For sequential hyperparameters, we consider their order matters.
        for name, hyperparameter_description in primitive.hyperparams.items():
            if only_control_hyperparams and hyperparameters is not None and CONTROL_HYPERPARAMETER_SEMANTIC_TYPE not in hyperparameters[name].semantic_types:
                continue
            is_set = isinstance(hyperparameters[name], hyperparams_module.Set) if hyperparameters is not None else False
            if hyperparameter_description['type'] == metadata_base.ArgumentType.DATA:
                if utils.is_sequence(hyperparameter_description['data']):
                    data_references: typing.List[str] = typing.cast(typing.List[str], hyperparameter_description['data'])
                    if is_set and not strict_order:
                        data_references = sorted(data_references)
                    for i, data_reference in enumerate(data_references):
                        self._inputs_ref[name, metadata_base.ArgumentType.DATA.name, i] = data_reference
                else:
                    self._inputs_ref[name, metadata_base.ArgumentType.DATA.name] = hyperparameter_description['data']
            elif hyperparameter_description['type'] == metadata_base.ArgumentType.PRIMITIVE:
                if utils.is_sequence(hyperparameter_description['data']):
                    primitive_references: typing.List[int] = typing.cast(typing.List[int], hyperparameter_description['data'])
                    if is_set and not strict_order:
                        primitive_references = sorted(primitive_references)
                    for i, primitive_reference in enumerate(primitive_references):
                        self._steps_ref[name, i] = primitive_reference
                else:
                    self._steps_ref[name] = hyperparameter_description['data']
            elif hyperparameter_description['type'] == metadata_base.ArgumentType.CONTAINER:
                self._inputs_ref[name, metadata_base.ArgumentType.CONTAINER.name] = hyperparameter_description['data']
            elif hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE:
                data = hyperparameter_description['data']
                if is_set and not strict_order:
                    assert isinstance(data, list)
                    # encode the value
                    simple_data = self._serialize_hyperparamter_value(name, data, True)
                    assert utils.is_sequence(simple_data)
                    data = [x for _, x in sorted(zip(simple_data, data), key=lambda pair: pair[0])]
                self.values[name] = data
            else:
                raise exceptions.UnexpectedValueError("Unknown hyper-parameter type: {hyperparameter_type}".format(hyperparameter_type=hyperparameter_description['type']))

    def reference_name(self) -> int:
        return self.index

    def output_reference_names(self) -> typing.List[str]:
        """
        The names for other nodes to refer its outputs.
        """

        return ['steps.{i}.{output_id}'.format(i=self.index, output_id=output_id) for output_id in self._outputs]

    def resolve_step_references(self, nodes_reverse_dict: typing.Dict[int, OrderedNode]) -> None:
        """
        Resolve step references with a lookup dict.
        """

        for input_index, ref in self._steps_ref.items():
            self.steps[input_index] = nodes_reverse_dict[ref]

    def _serialize_hyperparamter_value(self, name: str, data: typing.Any, is_sequence: bool) -> typing.Any:
        if self.primitive_step.primitive is not None:
            configuration = self.primitive_step.get_primitive_hyperparams().configuration
            if name not in configuration:
                raise exceptions.InvalidArgumentValueError(
                    "Unknown hyper-parameter name '{name}' for primitive {primitive}.".format(
                        name=name,
                        primitive=self.primitive_step.primitive,
                    ),
                )
            hyperparameter = configuration[name]
        else:
            hyperparameter = hyperparams_module.Hyperparameter[type(data)](data)  # type: ignore

        serialized = hyperparameter.value_to_json_structure(data)

        if is_sequence:
            return [json.dumps(s, sort_keys=True) for s in serialized]
        else:
            return json.dumps(serialized, sort_keys=True)

    def _unique_serialized_values(self) -> typing.Tuple:
        values = [(name, self._serialize_hyperparamter_value(name, data, False)) for name, data in self.values.items()]
        # Sort by value names.
        values.sort()
        return tuple(values)

    def _unique_step_references(self) -> typing.Tuple:
        steps_orders = [(name, node.global_order) for name, node in self.steps.items()]
        steps_orders.sort()
        return tuple(steps_orders)

    def unique_equivalence_class_repr(self) -> typing.Tuple:
        """
        Get the unique representation of the equivalence class of the node in the sense of isomorphism.
        """

        if not self.frozen or self._unique_equivalence_class_repr is None:
            repr_tuple = (self.name, self._unique_ordered_inputs(), self._unique_step_references(), self._unique_serialized_values(), self.topological_order)
            if self.frozen:
                self._unique_equivalence_class_repr = repr_tuple
            else:
                self._unique_equivalence_class_repr = None
            return repr_tuple

        return self._unique_equivalence_class_repr


class PlaceholderNode(OrderedNode):
    """
    This class represents a placeholder step in a DAG.

    Attributes
    ----------
    index:
        The index of this step in the pipeline.
    """

    index: int

    def __init__(self, placeholder: PlaceholderStep) -> None:
        super().__init__(PlaceholderStep.__name__, inputs_ref=placeholder.inputs.copy())
        self.index: int = placeholder.index
        self._outputs: typing.List[str] = placeholder.outputs.copy()

    def reference_name(self) -> int:
        return self.index

    def output_reference_names(self) -> typing.List[str]:
        """
        The names for other nodes to refer its outputs.
        """

        return ['steps.{i}.{output_id}'.format(i=self.index, output_id=output_id) for output_id in self._outputs]


class SubpipelineNode(OrderedNode):
    """
    This class represents a subpipeline step in a DAG.

    If this sub-pipeline has been resolved, then its graph is expected to be merged into its parent graph;
    otherwise `unique_equivalence_class_repr()` is called to get a unique representation according to its ID.

    Parameters
    ----------
    subpipeline:
        A subpipeline instance.

    Attributes
    ----------
    index:
        The index of this step in the pipeline.
    pipeline_id:
        The pipeline ID of subpipeline.
    pipeline:
        The sub-pipeline instance. If the sub-pipeline hasn't been resolved, it should be `None`.
    strict_order:
        If true, we will treat inputs of `Set` hyperparameters as a list.
        Otherwise we will try to sort their contents so the orders of their contents are not important.
    only_control_hyperparams:
        If true, hyperparameters that are not of the ``ControlParameter`` semantic type will not be included
        in the graph representation of this subpipeline's primitive steps.
    """

    index: int
    pipeline_id: str
    pipeline: typing.Optional[Pipeline]
    strict_order: bool
    only_control_hyperparams: bool

    def __init__(self, subpipeline: SubpipelineStep, *, strict_order: bool, only_control_hyperparams: bool) -> None:
        super().__init__(SubpipelineStep.__name__, inputs_ref=subpipeline.inputs.copy())
        self.strict_order = strict_order
        self.only_control_hyperparams = only_control_hyperparams
        self.index: int = subpipeline.index

        assert subpipeline.outputs is not None

        self._outputs: typing.List[str] = subpipeline.outputs.copy()
        self.pipeline_id: str = subpipeline.get_pipeline_id()
        self.pipeline: typing.Optional[Pipeline] = subpipeline.pipeline

    def graph(self) -> typing.Optional['PipelineDAG']:
        """
        Get the graph of the pipeline inside.

        Returns
        -------
        If this node has been resolved, return the graph; return None otherwise.
        """

        if self.pipeline is not None:
            return PipelineDAG(self.pipeline, strict_order=self.strict_order, only_control_hyperparams=self.only_control_hyperparams)
        return None

    def reference_name(self) -> int:
        return self.index

    def output_reference_names(self) -> typing.List[str]:
        """
        The names for other nodes to refer its outputs.
        """

        # Do not export null output_id.
        return ['steps.{i}.{output_id}'.format(i=self.index, output_id=output_id) for output_id in self._outputs if output_id is not None]

    def unique_equivalence_class_repr(self) -> typing.Tuple:
        """
        Get the unique representation of the equivalence class of the node in the sense of isomorphism.

        This is only used when the sub-pipeline hasn't been resolved. Otherwise, its graph should be used.
        """
        return super().unique_equivalence_class_repr() + (self.pipeline_id,)


class PipelineDAG:
    """
    Directed acyclic graph builder for a pipeline.

    It has an input node as the head of the DAG and an output node as the tail.

    Attributes
    ----------
    pipeline:
        The associated pipeline instance.
    step_nodes:
        These nodes belong to the steps of the pipeline, ordered by their index (including the extra inputs node & outputs node).
        It will be changed if we try to expand this graph.
    nodes:
        A set of **all** nodes in the graph.
        It will be changed if we try to expand this graph.
    strict_order:
        If true, we will treat inputs of `Set` hyperparameters as a list.
        Otherwise we will try to sort their contents so the orders of their contents are not important.
    only_control_hyperparams:
        If true, hyperparameters that are not of the ``ControlParameter`` semantic type will not be included
        in the graph representation of this pipeline's primitive steps.
    """

    pipeline: Pipeline
    step_nodes: typing.List[OrderedNode]
    nodes: typing.Set[OrderedNode]
    strict_order: bool
    only_control_hyperparams: bool

    def __init__(self, pipeline: Pipeline, *, strict_order: bool, only_control_hyperparams: bool) -> None:
        self.pipeline = pipeline
        self.strict_order = strict_order
        self.only_control_hyperparams = only_control_hyperparams

        self.step_nodes: typing.List[OrderedNode] = []
        self._nodes_reverse_dict: typing.Dict[int, OrderedNode] = {}
        self._nodes_outputs_reverse_dict: typing.Dict[str, typing.Tuple[OrderedNode, OutputIndex]] = {}

        self.inputs_node = InputsNode(pipeline.inputs)
        self.outputs_node = OutputsNode(pipeline.outputs)

        self.step_nodes.append(self.inputs_node)
        self.step_nodes.extend(self._convert_step_to_node(step) for step in pipeline.steps)
        self.step_nodes.append(self.outputs_node)

        self.nodes: typing.Set[OrderedNode] = set(self.step_nodes)

        # Build reversed mappings.
        for node in self.step_nodes:
            self._update_references(node)

        # Build the DAG.
        for node in self.step_nodes:
            self._resolve_references(node)

    def _convert_step_to_node(self, step: StepBase) -> OrderedNode:
        node: OrderedNode
        if isinstance(step, PrimitiveStep):
            node = PrimitiveNode(step, strict_order=self.strict_order, only_control_hyperparams=self.only_control_hyperparams)
        elif isinstance(step, PlaceholderStep):
            node = PlaceholderNode(step)
        elif isinstance(step, SubpipelineStep):
            node = SubpipelineNode(step, strict_order=self.strict_order, only_control_hyperparams=self.only_control_hyperparams)
        else:
            # New type of steps should be added here.
            raise NotImplementedError("Step type={t} is not supported.".format(t=type(step)))
        return node

    def _update_references(self, node: OrderedNode) -> None:
        for output_index, output_id in enumerate(node.output_reference_names()):
            self._nodes_outputs_reverse_dict[output_id] = (node, output_index)
        self._nodes_reverse_dict[node.reference_name()] = node

    def _resolve_references(self, node: OrderedNode) -> None:
        node.resolve_input_references(self._nodes_outputs_reverse_dict)
        if isinstance(node, PrimitiveNode):
            node.resolve_step_references(self._nodes_reverse_dict)

    def body_nodes(self) -> typing.Set[OrderedNode]:
        """
        Return all nodes expect the inputs node and outputs node in the graph.
        """

        return self.nodes - {self.inputs_node, self.outputs_node}

    def expand_node(self, node: OrderedNode, graph: PD) -> None:
        """
        Replace a node with a graph.
        """

        assert node in self.nodes

        # Update node records.
        loc = self.step_nodes.index(node)
        self.step_nodes = self.step_nodes[:loc] + graph.step_nodes[1:-1] + self.step_nodes[loc + 1:]
        self.nodes.remove(node)
        self.nodes.update(graph.body_nodes())

        # Join nodes.
        graph.inputs_node.join(node)
        node.join(graph.outputs_node)

    def expand_subpipelines(self, recursive: bool = True) -> None:
        """
        Extract all nodes inside a subpipeline's graph and integrate them into this graph.

        Parameters
        ----------
        recursive:
            If true, we will expand subpipelines of all depth (that is, subpipelines of subpipelines).
        """

        # Pick up subpipeline nodes into a list because expanding nodes will change the graph.
        subpipelines: typing.List[SubpipelineNode] = [node for node in self.nodes if isinstance(node, SubpipelineNode)]
        for subpipeline_node in subpipelines:
            subgraph: typing.Optional[PipelineDAG] = subpipeline_node.graph()
            if subgraph is not None:
                if recursive:
                    subgraph.expand_subpipelines(recursive=recursive)
                self.expand_node(subpipeline_node, subgraph)


class PipelineHasher:
    """
    Hash helper for pipelines.

    This algorithm checks if the two pipelines are equal in the sense of isomorphism by solving a graph isomorphism
    problem. The general graph isomorphism problem is known to be neither P nor NP-complete. However,
    our pipelines are DAGs so we could have an algorithm to check its isomorphism in polynomial time.

    The complexity of this algorithm is around :math:`O((V + E)logV)`, where :math:`V` is the number of steps in the
    pipeline and :math:`E` is the number of output references.

    The algorithm follows these steps:

    1. Construct a DAG from the given pipeline. A directed edge is pointed from A to B if A depends on B directly.
    2. Perform topological sort on the DAG using DFS. Nodes with same topological order are put into the same layer.
    3. Using a greedy algorithm to get 'global' orders of nodes.
       It sorts the nodes in the same layer by making use of the global order of nodes they depend on.
    4. Get a unique, hashable & comparable tuple representing the structure of the pipeline according to the global order of nodes.
       It also provides a unique representation of the equivalence class of a pipeline in the sense of isomorphism.

    And about supporting new steps, one should extend PipelineDAG._convert_step_to_node`.

    Attributes
    ----------
    pipeline:
        The associated pipeline instance.
    graph:
        The graph representation of the pipeline.
    strict_order:
        If true, we will treat inputs of `Set` hyperparameters as a list, and the order of primitives are determined by their step indices.
        Otherwise we will try to sort contents of `Set` hyperparameters so the orders of their contents are not important,
        and we will try topological sorting to determine the order of nodes.
    """

    pipeline: Pipeline
    graph: PipelineDAG
    strict_order: bool

    def __init__(self, pipeline: Pipeline, strict_order: bool = False, only_control_hyperparams: bool = False) -> None:
        self.pipeline = pipeline
        self.strict_order = strict_order
        self.graph = PipelineDAG(pipeline, strict_order=strict_order, only_control_hyperparams=only_control_hyperparams)
        self.graph.expand_subpipelines(recursive=True)

        self._hash: typing.Optional[int] = None
        self._representation: typing.Optional[typing.Tuple] = None
        self._layers: typing.List[typing.List[OrderedNode]] = [[self.graph.inputs_node]]

        self._unordered_nodes: typing.Set[OrderedNode] = set()

    def _dfs_topological_ordering(self, node: OrderedNode) -> OrderedNode:
        for parent, output_index in node.inputs.values():
            if parent in self._unordered_nodes:
                self._dfs_topological_ordering(parent)
            node.topological_order = max(node.topological_order, parent.topological_order + 1)

        self._unordered_nodes.remove(node)

        # Classify it into layers.
        while len(self._layers) < node.topological_order + 1:
            self._layers.append([])
        self._layers[node.topological_order].append(node)

        return node

    def _global_ordering(self) -> None:
        global_order = -1
        for layer in self._layers:
            for node in layer:
                node.frozen = True  # Enable cache so we can be much faster in comparison.
            layer.sort(key=lambda x: x.unique_equivalence_class_repr())
            last = None
            for j, node in enumerate(layer):
                # Keep symmetric. Nodes with same local_order should have same global_order.
                if node.unique_equivalence_class_repr() != last:
                    global_order += 1
                    last = node.unique_equivalence_class_repr()
                node.global_order = global_order

    def unique_equivalence_class_repr(self) -> typing.Tuple:
        """
        Get the unique representation of the equivalence class of the pipeline in the sense of isomorphism.
        """

        if self._representation is None:
            if self.strict_order:
                for i, node in enumerate(self.graph.step_nodes):
                    node.topological_order = i
                    node.global_order = i
                self._representation = tuple(node.unique_equivalence_class_repr() for node in self.graph.step_nodes)
            else:
                self._unordered_nodes = self.graph.nodes.copy()
                self._unordered_nodes.remove(self.graph.inputs_node)
                # Perform topological sort.
                while self._unordered_nodes:
                    node = next(iter(self._unordered_nodes))  # Retrieve an item without deleting it.
                    self._dfs_topological_ordering(node)

                self._global_ordering()
                self._representation = tuple(node.unique_equivalence_class_repr() for layer in self._layers for node in layer)

        return self._representation

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.unique_equivalence_class_repr())
        return self._hash


def get_pipeline(
    pipeline_path: str, *, strict_resolving: bool = False, strict_digest: bool = False,
    pipeline_search_paths: typing.Sequence[str] = None, respect_environment_variable: bool = True, load_all_primitives: bool = True,
    resolver_class: typing.Type[Resolver] = Resolver, pipeline_class: typing.Type[Pipeline] = Pipeline,
) -> Pipeline:
    resolver = resolver_class(
        strict_resolving=strict_resolving, strict_digest=strict_digest, pipeline_search_paths=pipeline_search_paths,
        respect_environment_variable=respect_environment_variable, load_all_primitives=load_all_primitives,
    )

    if os.path.exists(pipeline_path):
        with utils.open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
            if pipeline_path.endswith('.yml') or pipeline_path.endswith('.yaml'):
                return pipeline_class.from_yaml(pipeline_file, resolver=resolver, strict_digest=strict_digest)
            elif pipeline_path.endswith('.json'):
                return pipeline_class.from_json(pipeline_file, resolver=resolver, strict_digest=strict_digest)
            else:
                raise ValueError("Unknown file extension.")
    else:
        return resolver.get_pipeline({'id': pipeline_path})


def describe_handler(
    arguments: argparse.Namespace, *, resolver_class: typing.Type[Resolver] = None,
    no_resolver_class: typing.Type[Resolver] = None, pipeline_class: typing.Type[Pipeline] = None,
) -> None:
    if resolver_class is None:
        resolver_class = Resolver
    if no_resolver_class is None:
        no_resolver_class = NoResolver
    if pipeline_class is None:
        pipeline_class = Pipeline

    if getattr(arguments, 'no_resolving', False):
        resolver: Resolver = no_resolver_class()
    else:
        resolver = resolver_class(
            strict_resolving=getattr(arguments, 'strict_resolving', False),
            strict_digest=getattr(arguments, 'strict_digest', False),
            pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
        )

    output_stream = getattr(arguments, 'output', sys.stdout)

    has_errored = False

    for pipeline_path in arguments.pipelines:
        if getattr(arguments, 'list', False):
            print(pipeline_path, file=output_stream)

        try:
            with utils.open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
                if pipeline_path.endswith('.yml') or pipeline_path.endswith('.yaml') or pipeline_path.endswith('.yml.gz') or pipeline_path.endswith('.yaml.gz'):
                    pipeline = pipeline_class.from_yaml(
                        pipeline_file,
                        resolver=resolver,
                        strict_digest=getattr(arguments, 'strict_digest', False),
                    )
                elif pipeline_path.endswith('.json') or pipeline_path.endswith('.json.gz'):
                    pipeline = pipeline_class.from_json(
                        pipeline_file,
                        resolver=resolver,
                        strict_digest=getattr(arguments, 'strict_digest', False),
                    )
                else:
                    raise ValueError("Unknown file extension.")
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error parsing pipeline: {pipeline_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error parsing pipeline: {pipeline_path}") from error

        if getattr(arguments, 'check', True):
            try:
                pipeline.check(
                    allow_placeholders=getattr(arguments, 'allow_placeholders', False),
                    standard_pipeline=getattr(arguments, 'standard_pipeline', True),
                )
            except Exception as error:
                if getattr(arguments, 'continue', False):
                    traceback.print_exc(file=output_stream)
                    print(f"Error checking pipeline: {pipeline_path}", file=output_stream)
                    has_errored = True
                    continue
                else:
                    raise Exception("Error checking pipeline: {pipeline_path}".format(pipeline_path=pipeline_path)) from error

        try:
            if getattr(arguments, 'set_source_name', None) is not None:
                if pipeline.source is None:
                    pipeline.source = {}
                if arguments.set_source_name:
                    pipeline.source['name'] = arguments.set_source_name
                elif 'name' in pipeline.source:
                    del pipeline.source['name']
                if not pipeline.source:
                    pipeline.source = None

            pipeline_description = pipeline.to_json_structure(canonical=True)

            if getattr(arguments, 'print', False):
                pprint.pprint(pipeline_description, stream=output_stream)
            else:
                json.dump(
                    pipeline_description,
                    output_stream,
                    indent=(getattr(arguments, 'indent', 2) or None),
                    sort_keys=getattr(arguments, 'sort_keys', False),
                    allow_nan=False,
                )  # type: ignore
                output_stream.write('\n')
        except Exception as error:
            if getattr(arguments, 'continue', False):
                traceback.print_exc(file=output_stream)
                print(f"Error describing pipeline: {pipeline_path}", file=output_stream)
                has_errored = True
                continue
            else:
                raise Exception(f"Error describing pipeline: {pipeline_path}") from error

    if has_errored:
        sys.exit(1)


if pyarrow_lib is not None:
    pyarrow_lib._default_serialization_context.register_type(
        Pipeline, 'd3m.pipeline', pickle=True,
    )


def main(argv: typing.Sequence) -> None:
    raise exceptions.NotSupportedError("This CLI has been removed. Use \"python3 -m d3m pipeline describe\" instead.")


if __name__ == '__main__':
    main(sys.argv)
