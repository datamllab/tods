import abc
import argparse
import base64
import collections
import contextlib
import copy
import datetime
import decimal
import enum
import functools
import gzip
import hashlib
import inspect
import json
import logging
import numbers
import operator
import os
import os.path
import pickle
import random
import re
import types
import typing
import sys
import unittest
import uuid
from urllib import parse as url_parse

import custom_inherit  # type: ignore
import frozendict  # type: ignore
import git  # type: ignore
import jsonpath_ng  # type: ignore
import jsonschema  # type: ignore
import numpy  # type: ignore
import pandas  # type: ignore
import typing_inspect  # type: ignore
import yaml  # type: ignore
import pyrsistent  # type: ignore
from jsonschema import validators  # type: ignore
from numpy import random as numpy_random  # type: ignore
from pytypes import type_util  # type: ignore

import d3m
from d3m import deprecate, exceptions

if yaml.__with_libyaml__:
    from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper  # type: ignore
else:
    from yaml import SafeLoader, SafeDumper

logger = logging.getLogger(__name__)

NONE_TYPE: typing.Type = type(None)

# Only types without elements can be listed here. If they are elements, we have to
# check all elements as well.
KNOWN_IMMUTABLE_TYPES = (
    str, int, float, bool, numbers.Integral, decimal.Decimal,
    numbers.Real, numpy.integer, numpy.float32, numpy.float64, numpy.bool_, bytes,
    datetime.date, datetime.time, datetime.datetime, NONE_TYPE, enum.Enum,
)

HASH_ID_NAMESPACE = uuid.UUID('8614b2cc-89ef-498e-9254-833233b3959b')

PACKAGE_BASE = os.path.dirname(d3m.__file__)


def current_git_commit(path: str, search_parent_directories: bool = True) -> str:
    """
    Returns a git commit hash of the repo at ``path`` or above if ``search_parent_directories`` is ``True``.

    When used to get a commit hash of a Python package, for this to work, the package has
    to be installed in "editable" mode (``pip install -e``).

    Parameters
    ----------
    path:
        A path to repo or somewhere under the repo.
    search_parent_directories:
        Whether to search for a git repository in directories above ``path``.

    Returns
    -------
    A git commit hash.
    """

    repo = git.Repo(path=path, search_parent_directories=search_parent_directories)
    return repo.head.object.hexsha


# Using typing.TypeVar in type signature does not really work, so we are using type instead.
# See: https://github.com/python/typing/issues/520
def get_type_arguments(cls: type, *, unique_names: bool = False) -> typing.Dict[type, type]:
    """
    Returns a mapping between type arguments and their types of a given class ``cls``.

    Parameters
    ----------
    cls:
        A class to return mapping for.
    unique_names:
        Should we force unique names of type parameters.

    Returns
    -------
    A mapping from type argument to its type.
    """

    # Using typing.TypeVar in type signature does not really work, so we are using type instead.
    # See: https://github.com/python/typing/issues/520
    result: typing.Dict[type, type] = {}

    for base_class in inspect.getmro(typing_inspect.get_origin(cls)):
        if base_class == typing.Generic:
            break

        if not typing_inspect.is_generic_type(base_class):
            continue

        parameters = typing_inspect.get_parameters(base_class)

        # We are using _select_Generic_superclass_parameters and not get_Generic_parameters
        # so that we can handle the case where the result is None.
        # See: https://github.com/Stewori/pytypes/issues/20
        arguments = type_util._select_Generic_superclass_parameters(cls, base_class)

        if arguments is None:
            arguments = [typing.Any] * len(parameters)

        if len(parameters) != len(arguments):
            raise TypeError("Number of parameters does not match number of arguments.")

        for parameter, argument in zip(parameters, arguments):
            if type_util.resolve_fw_decl(argument, module_name=base_class.__module__, globs=dir(sys.modules[base_class.__module__]))[1]:
                argument = argument.__forward_value__

            visited: typing.Set[type] = set()
            while typing_inspect.is_typevar(argument) and argument in result:
                if argument in visited:
                    raise RuntimeError("Loop while resolving type variables.")
                visited.add(argument)

                argument = result[argument]

            if parameter == argument:
                argument = typing.Any

            if parameter in result:
                if result[parameter] != argument:
                    raise TypeError("Different types for same parameter across class bases: {type1} vs. {type2}".format(
                        type1=result[parameter],
                        type2=argument,
                    ))
            else:
                result[parameter] = argument

    if unique_names:
        type_parameter_names = [parameter.__name__ for parameter in result.keys()]

        type_parameter_names_set = set(type_parameter_names)

        if len(type_parameter_names) != len(type_parameter_names_set):
            for name in type_parameter_names_set:
                type_parameter_names.remove(name)
            raise TypeError("Same name reused across different type parameters: {extra_names}".format(extra_names=type_parameter_names))

    return result


def is_instance(obj: typing.Any, cls: typing.Union[type, typing.Tuple[type]]) -> bool:
    # We do not want really to check generators. A workaround.
    # See: https://github.com/Stewori/pytypes/issues/49
    if isinstance(obj, types.GeneratorType):
        return False

    if isinstance(cls, tuple):
        cls = typing.Union[cls]  # type: ignore

    # "bound_typevars" argument has to be passed for this function to
    # correctly work with type variables.
    # See: https://github.com/Stewori/pytypes/issues/24
    return type_util._issubclass(type_util.deep_type(obj), cls, bound_typevars={})


def is_subclass(subclass: type, superclass: typing.Union[type, typing.Tuple[type]]) -> bool:
    # "bound_typevars" argument has to be passed for this function to
    # correctly work with type variables.
    # See: https://github.com/Stewori/pytypes/issues/24
    return type_util._issubclass(subclass, superclass, bound_typevars={})


def get_type(obj: typing.Any) -> type:
    typ = type_util.deep_type(obj, depth=1)

    if is_subclass(typ, type_util.Empty):
        typ = typing_inspect.get_last_args(typ)[0]

    return typ


def is_instance_method_on_class(method: typing.Any) -> bool:
    if is_class_method_on_class(method):
        return False

    if inspect.isfunction(method):
        return True

    if getattr(method, '__func__', None):
        return True

    return False


def is_class_method_on_class(method: typing.Any) -> bool:
    return inspect.ismethod(method)


def is_instance_method_on_object(method: typing.Any, object: typing.Any) -> bool:
    if not inspect.ismethod(method):
        return False

    if method.__self__ is object:
        return True

    return False


def is_class_method_on_object(method: typing.Any, object: typing.Any) -> bool:
    if not inspect.ismethod(method):
        return False

    if method.__self__ is type(object):
        return True

    return False


def is_type(obj: typing.Any) -> bool:
    return isinstance(obj, type) or obj is typing.Any or typing_inspect.is_tuple_type(obj) or typing_inspect.is_union_type(obj)


def type_to_str(obj: type) -> str:
    return type_util.type_str(obj, assumed_globals={}, update_assumed_globals=False)


def get_type_hints(func: typing.Callable) -> typing.Dict[str, typing.Any]:
    # To skip decorators. Same stop function as used in "inspect.signature".
    func = inspect.unwrap(func, stop=(lambda f: hasattr(f, '__signature__')))
    return type_util.get_type_hints(func)


yaml_warning_issued = False


def yaml_dump_all(documents: typing.Sequence[typing.Any], stream: typing.IO[typing.Any] = None, **kwds: typing.Any) -> typing.Any:
    global yaml_warning_issued

    if not yaml.__with_libyaml__ and not yaml_warning_issued:
        yaml_warning_issued = True
        logger.warning("cyaml not found, using a slower pure Python YAML implementation.")

    return yaml.dump_all(documents, stream, Dumper=SafeDumper, **kwds)


def yaml_dump(data: typing.Any, stream: typing.IO[typing.Any] = None, **kwds: typing.Any) -> typing.Any:
    global yaml_warning_issued

    if not yaml.__with_libyaml__ and not yaml_warning_issued:
        yaml_warning_issued = True
        logger.warning("cyaml not found, using a slower pure Python YAML implementation.")

    return yaml.dump_all([data], stream, Dumper=SafeDumper, **kwds)


def yaml_load_all(stream: typing.Union[str, typing.IO[typing.Any]]) -> typing.Any:
    global yaml_warning_issued

    if not yaml.__with_libyaml__ and not yaml_warning_issued:
        yaml_warning_issued = True
        logger.warning("cyaml not found, using a slower pure Python YAML implementation.")

    return yaml.load_all(stream, SafeLoader)


def yaml_load(stream: typing.Union[str, typing.IO[typing.Any]]) -> typing.Any:
    global yaml_warning_issued

    if not yaml.__with_libyaml__ and not yaml_warning_issued:
        yaml_warning_issued = True
        logger.warning("cyaml not found, using a slower pure Python YAML implementation.")

    return yaml.load(stream, SafeLoader)


def yaml_add_representer(value_type: typing.Type, represented: typing.Callable) -> None:
    yaml.Dumper.add_representer(value_type, represented)
    yaml.SafeDumper.add_representer(value_type, represented)

    if yaml.__with_libyaml__:
        yaml.CDumper.add_representer(value_type, represented)  # type: ignore
        yaml.CSafeDumper.add_representer(value_type, represented)  # type: ignore


class EnumMeta(enum.EnumMeta):
    def __new__(mcls, class_name, bases, namespace, **kwargs):  # type: ignore
        def __reduce_ex__(self: typing.Any, proto: int) -> typing.Any:
            return self.__class__, (self._value_,)

        if '__reduce_ex__' not in namespace:
            namespace['__reduce_ex__'] = __reduce_ex__

        cls = super().__new__(mcls, class_name, bases, namespace, **kwargs)

        def yaml_representer(dumper, data):  # type: ignore
            return yaml.ScalarNode('tag:yaml.org,2002:str', data.name)

        yaml_add_representer(cls, yaml_representer)

        return cls


class Enum(enum.Enum, metaclass=EnumMeta):
    """
    An extension of `Enum` base class where:

    * Instances are equal to their string names, too.
    * It registers itself with "yaml" module to serialize itself as a string.
    * Allows dynamic registration of additional values using ``register_value``.
    """

    def __eq__(self, other):  # type: ignore
        if isinstance(other, str):
            return self.name == other

        return super().__eq__(other)

    # It must hold a == b => hash(a) == hash(b). Because we allow enums to be equal to names,
    # the easiest way to assure the condition is to hash everything according to their names.
    def __hash__(self):  # type: ignore
        return hash(self.name)

    @classmethod
    def register_value(cls, name: str, value: typing.Any) -> typing.Any:
        # This code is based on Python's "EnumMeta.__new__" code, see
        # comments there for more information about the code.
        # It uses internals of Python's Enum so it is potentially fragile.

        __new__, save_new, use_args = type(cls)._find_new_({}, cls._member_type_, cls)  # type: ignore

        dynamic_attributes = {
            k for c in cls.mro()
            for k, v in c.__dict__.items()
            if isinstance(v, types.DynamicClassAttribute)
        }

        if not isinstance(value, tuple):
            args: typing.Tuple[typing.Any, ...] = (value,)
        else:
            args = value
        if cls._member_type_ is tuple:  # type: ignore
            args = (args,)

        if not use_args:
            enum_member = __new__(cls)
            if not hasattr(enum_member, '_value_'):
                enum_member._value_ = value
        else:
            enum_member = __new__(cls, *args)
            if not hasattr(enum_member, '_value_'):
                if cls._member_type_ is object:  # type: ignore
                    enum_member._value_ = value
                else:
                    enum_member._value_ = cls._member_type_(*args)  # type: ignore
        value = enum_member._value_
        enum_member._name_ = name
        enum_member.__objclass__ = cls
        enum_member.__init__(*args)
        for canonical_member in cls._member_map_.values():  # type: ignore
            if canonical_member._value_ == enum_member._value_:
                enum_member = canonical_member
                break
        else:
            cls._member_names_.append(name)  # type: ignore
        if name not in dynamic_attributes:
            setattr(cls, name, enum_member)
        cls._member_map_[name] = enum_member  # type: ignore
        try:
            cls._value2member_map_[value] = enum_member  # type: ignore
        except TypeError:
            pass


# Return type has to be "Any" because mypy does not support enums generated dynamically
# and complains about missing attributes when trying to access them.
def create_enum_from_json_schema_enum(
    class_name: str, obj: typing.Dict, json_paths: typing.Union[typing.Sequence[str], str], *,
    module: str = None, qualname: str = None, base_class: type = None
) -> typing.Any:
    if qualname is None:
        qualname = class_name

    if isinstance(json_paths, str):
        names = _get_names(obj, json_paths)
    else:
        names = []
        for path in json_paths:
            names += _get_names(obj, path)

    # Make the list contain unique names. It keeps the original order in Python 3.6+
    # because dicts are ordered. We use the same string for both the name and the value.
    pairs = [(name, name) for name in dict.fromkeys(names).keys()]

    return Enum(value=class_name, names=pairs, module=module, qualname=qualname, type=base_class)  # type: ignore


def _get_names(obj: typing.Dict, path: str) -> typing.List:
    json_path_expression = jsonpath_ng.parse(path)
    return [match.value for match in json_path_expression.find(obj)]


# This allows other modules to register additional immutable values and types.
# We are doing it this way to overcome issues with import cycles.
additional_immutable_values: typing.Tuple[typing.Any, ...] = ()
additional_immutable_types: typing.Tuple[type, ...] = ()


def make_immutable_copy(obj: typing.Any) -> typing.Any:
    """
    Converts a given ``obj`` into an immutable copy of it, if possible.

    Parameters
    ----------
    obj:
        Object to convert.

    Returns
    -------
    An immutable copy of ``obj``.
    """

    if any(obj is immutable_value for immutable_value in additional_immutable_values):
        return obj

    if isinstance(obj, numpy.matrix):
        # One cannot iterate over a matrix segment by segment. You always get back
        # a matrix (2D structure) and not an array of rows or columns. By converting
        # it to an array such iteration segment by segment works.
        obj = numpy.array(obj)

    if isinstance(obj, KNOWN_IMMUTABLE_TYPES):
        # Because str is among known immutable types, it will not be picked apart as a sequence.
        return obj
    if additional_immutable_types and isinstance(obj, additional_immutable_types):
        return obj
    if is_type(obj):
        # Assume all types are immutable.
        return obj
    if isinstance(obj, typing.Mapping):
        # We simply always preserve order of the mapping. Because we want to make sure also mapping's
        # values are converted to immutable values, we cannot simply use MappingProxyType.
        return frozendict.FrozenOrderedDict((make_immutable_copy(k), make_immutable_copy(v)) for k, v in obj.items())
    if isinstance(obj, typing.Set):
        return frozenset(make_immutable_copy(o) for o in obj)
    if isinstance(obj, tuple):
        # To preserve named tuples.
        return type(obj)(make_immutable_copy(o) for o in obj)
    if isinstance(obj, pandas.DataFrame):
        return tuple(make_immutable_copy(o) for o in obj.itertuples(index=False, name=None))
    if isinstance(obj, (typing.Sequence, numpy.ndarray)):
        return tuple(make_immutable_copy(o) for o in obj)

    raise TypeError("{obj} is not known to be immutable.".format(obj=obj))


def check_immutable(obj: typing.Any) -> None:
    """
    Checks that ``obj`` is immutable. Raises an exception if this is not true.

    Parameters
    ----------
    obj:
        Object to check.
    """

    obj_type = type(obj)

    # First check common cases.
    if any(obj is immutable_value for immutable_value in additional_immutable_values):
        return
    if obj_type in KNOWN_IMMUTABLE_TYPES:
        return
    if obj_type is frozendict.FrozenOrderedDict:
        for k, v in obj.items():
            check_immutable(k)
            check_immutable(v)
        return
    if obj_type is tuple:
        for o in obj:
            check_immutable(o)
        return

    if isinstance(obj, KNOWN_IMMUTABLE_TYPES):
        return
    if additional_immutable_types and isinstance(obj, additional_immutable_types):
        return
    if isinstance(obj, tuple):
        # To support named tuples.
        for o in obj:
            check_immutable(o)
        return
    if is_type(obj):
        # Assume all types are immutable.
        return
    if obj_type is frozenset:
        for o in obj:
            check_immutable(o)
        return

    raise TypeError("{obj} is not known to be immutable.".format(obj=obj))


class Metaclass(custom_inherit._DocInheritorBase):
    """
    A metaclass which makes sure docstrings are inherited.

    It knows how to merge numpy-style docstrings and merge parent sections with
    child sections. For example, then it is not necessary to repeat documentation
    for parameters if they have not changed.
    """

    @staticmethod
    def class_doc_inherit(prnt_doc: str = None, child_doc: str = None) -> typing.Optional[str]:
        return custom_inherit.store['numpy'](prnt_doc, child_doc)

    @staticmethod
    def attr_doc_inherit(prnt_doc: str = None, child_doc: str = None) -> typing.Optional[str]:
        return custom_inherit.store['numpy'](prnt_doc, child_doc)


class AbstractMetaclass(abc.ABCMeta, Metaclass):
    """
    A metaclass which makes sure docstrings are inherited. For use with abstract classes.
    """


class GenericMetaclass(typing.GenericMeta, Metaclass):
    """
    A metaclass which makes sure docstrings are inherited. For use with generic classes (which are also abstract).
    """


class RefResolverNoRemote(validators.RefResolver):
    def resolve_remote(self, uri: str) -> typing.Any:
        raise exceptions.NotSupportedError("Remote resolving disabled: {uri}".format(uri=uri))


def enum_validator(validator, enums, instance, schema):  # type: ignore
    if isinstance(instance, Enum):
        instance = instance.name

    yield from validators.Draft7Validator.VALIDATORS['enum'](validator, enums, instance, schema)


def json_schema_is_string(checker: jsonschema.TypeChecker, instance: typing.Any) -> bool:
    if isinstance(instance, Enum):
        return True
    else:
        return validators.Draft7Validator.TYPE_CHECKER.is_type(instance, 'string')


def json_schema_is_object(checker: jsonschema.TypeChecker, instance: typing.Any) -> bool:
    if isinstance(instance, (frozendict.frozendict, frozendict.FrozenOrderedDict)):
        return True
    else:
        return validators.Draft7Validator.TYPE_CHECKER.is_type(instance, 'object')


def json_schema_is_array(checker: jsonschema.TypeChecker, instance: typing.Any) -> bool:
    if isinstance(instance, (tuple, set)):
        return True
    else:
        return validators.Draft7Validator.TYPE_CHECKER.is_type(instance, 'array')


JsonSchemaTypeChecker = validators.Draft7Validator.TYPE_CHECKER.redefine_many({
    'string': json_schema_is_string,
    'object': json_schema_is_object,
    'array': json_schema_is_array,
})


# JSON schema validator with the following extension:
#
# * If a value is an instance of Python enumeration, its name is checked against JSON
#   schema enumeration, instead of the value itself. When converting to a proper JSON
#   these values should be enumeration's name.
Draft7Validator = validators.extend(
    validators.Draft7Validator,
    validators={
        'enum': enum_validator,
    },
    type_checker=JsonSchemaTypeChecker,
)


draft7_format_checker = copy.deepcopy(jsonschema.draft7_format_checker)


@draft7_format_checker.checks('python-type')
def json_schema_is_python_type(instance: typing.Any) -> bool:
    return is_type(instance) or isinstance(instance, str)


# We cannot use "Draft7Validator" as a type (MyPy complains), so we are using
# "validators.Draft7Validator", which has the same interface.
def load_schema_validators(schemas: typing.Dict, load_validators: typing.Sequence[str]) -> typing.List[validators.Draft7Validator]:
    schema_validators = []

    for schema_filename in load_validators:
        for schema_uri, schema_json in schemas.items():
            if os.path.basename(schema_uri) == schema_filename:
                break
        else:
            raise exceptions.InvalidArgumentValueError("Cannot find schema '{schema_filename}'.".format(schema_filename=schema_filename))

        # We validate schemas using unmodified validator.
        validators.Draft7Validator.check_schema(schema_json)

        validator = Draft7Validator(
            schema=schema_json,
            resolver=RefResolverNoRemote(schema_json['id'], schema_json, schemas),
            format_checker=draft7_format_checker,
        )

        schema_validators.append(validator)

    return schema_validators


def datetime_for_json(timestamp: datetime.datetime) -> str:
    # Since Python 3.6 "astimezone" can be called on naive instances
    # that are presumed to represent system local time.
    # We remove timezone information before formatting to not have "+00:00" added and
    # we then manually add "Z" instead (which has equivalent meaning).
    return timestamp.astimezone(datetime.timezone.utc).replace(tzinfo=None).isoformat('T') + 'Z'


class JsonEncoder(json.JSONEncoder):
    """
    JSON encoder with extensions, among them the main ones are:

    * Frozen dict is encoded as a dict.
    * Python types are encoded into strings describing them.
    * Python enumerations are encoded into their string names.
    * Sets are encoded into lists.
    * Encodes ndarray and DataFrame as nested lists.
    * Encodes datetime into ISO format with UTC timezone.
    * Everything else which cannot be encoded is converted to a string.

    You probably want to use `to_json_structure` and not this class, because `to_json_structure`
    also encodes ``NaN`, ``Infinity``, and ``-Infinity`` as strings.

    It does not necessary make a JSON which can then be parsed back to reconstruct original value.
    """

    def default(self, o: typing.Any) -> typing.Any:
        # Importing here to prevent import cycle.
        from d3m.metadata import base

        if isinstance(o, numpy.matrix):
            # One cannot iterate over a matrix segment by segment. You always get back
            # a matrix (2D structure) and not an array of rows or columns. By converting
            # it to an array such iteration segment by segment works.
            o = numpy.array(o)

        if isinstance(o, frozendict.frozendict):
            return dict(o)
        if isinstance(o, frozendict.FrozenOrderedDict):
            return collections.OrderedDict(o)
        if is_type(o):
            return type_to_str(o)
        if isinstance(o, Enum):
            return o.name
        if o is base.ALL_ELEMENTS:
            return repr(o)
        if o is base.NO_VALUE:
            return repr(o)
        # For encoding numpy.int64, numpy.float64 already works.
        if isinstance(o, numpy.integer):
            return int(o)
        if isinstance(o, numpy.bool_):
            return bool(o)
        if isinstance(o, typing.Mapping):
            return collections.OrderedDict(o)
        if isinstance(o, typing.Set):
            return sorted(o, key=str)
        if isinstance(o, pandas.DataFrame):
            return list(o.itertuples(index=False, name=None))
        if isinstance(o, (typing.Sequence, numpy.ndarray)):
            return list(o)
        if isinstance(o, decimal.Decimal):
            return float(o)
        if isinstance(o, bytes):
            return base64.b64encode(o).decode('utf8')
        if isinstance(o, datetime.datetime):
            return datetime_for_json(o)

        try:
            return super().default(o)
        except TypeError:
            return str(o)


def normalize_numbers(obj: typing.Dict) -> typing.Dict:
    return json.loads(json.dumps(obj), parse_int=float)


json_constant_map = {
    '-Infinity': str(float('-Infinity')),
    'Infinity': str(float('Infinity')),
    'NaN': str(float('NaN')),
}


def to_json_structure(obj: typing.Any) -> typing.Any:
    """
    In addition to what `JsonEncoder` encodes, this function also encodes as strings
    float ``NaN``, ``Infinity``, and ``-Infinity``.

    It does not necessary make a JSON structure which can then be parsed back to reconstruct
    original value. For that use ``to_reversible_json_structure``.
    """

    # We do not use "allow_nan=False" here because we will handle those values during loading.
    # "JsonEncoder.default" is not called for float values so we cannot handle them there.
    # See: https://bugs.python.org/issue36841
    json_string = json.dumps(obj, cls=JsonEncoder)

    return json.loads(
        json_string,
        parse_constant=lambda constant: json_constant_map[constant],
    )


def _json_key(key: typing.Any) -> str:
    if isinstance(key, str):
        return key
    else:
        raise TypeError("Key must be a string, not '{key_type}'.".format(key_type=type(key)))


def to_reversible_json_structure(obj: typing.Any) -> typing.Any:
    """
    Operation is not idempotent.
    """

    if isinstance(obj, (str, bool, NONE_TYPE)):
        return obj

    obj_type = type(obj)

    if _is_int(obj_type):
        # To make sure it is Python int.
        obj = int(obj)

        return obj

    elif _is_float(obj_type):
        # To make sure it is Python float.
        obj = float(obj)

        if not numpy.isfinite(obj):
            return {
                'encoding': 'pickle',
                'description': str(obj),
                'value': base64.b64encode(pickle.dumps(obj)).decode('utf8'),
            }
        else:
            return obj

    elif isinstance(obj, typing.Mapping):
        if 'encoding' in obj and 'value' in obj:
            return {
                'encoding': 'escape',
                'value': {_json_key(k): to_reversible_json_structure(v) for k, v in obj.items()},
            }
        else:
            return {_json_key(k): to_reversible_json_structure(v) for k, v in obj.items()}

    # We do not use "is_sequence" because we do not want to convert all sequences,
    # because it can be loosing important information.
    elif isinstance(obj, (tuple, list)):
        return [to_reversible_json_structure(v) for v in obj]

    else:
        return {
            'encoding': 'pickle',
            'description': str(obj),
            'value': base64.b64encode(pickle.dumps(obj)).decode('utf8'),
        }


def from_reversible_json_structure(obj: typing.Any) -> typing.Any:
    if is_instance(obj, typing.Union[str, int, float, bool, NONE_TYPE]):
        return obj

    elif isinstance(obj, typing.Mapping):
        if 'encoding' in obj and 'value' in obj:
            if obj['encoding'] == 'pickle':
                # TODO: Limit the types of values being able to load to prevent arbitrary code execution by a malicious pickle.
                return pickle.loads(base64.b64decode(obj['value'].encode('utf8')))
            if obj['encoding'] == 'escape':
                return {_json_key(k): from_reversible_json_structure(v) for k, v in obj['value'].items()}
            else:
                raise ValueError("Unsupported encoding '{encoding}'.".format(encoding=obj['encoding']))
        else:
            return {_json_key(k): from_reversible_json_structure(v) for k, v in obj.items()}

    # We do not use "is_sequence" because we do not want to convert all sequences,
    # because it can be loosing important information.
    elif isinstance(obj, (tuple, list)):
        return [from_reversible_json_structure(v) for v in obj]

    else:
        raise TypeError("Unsupported type '{value_type}'.".format(value_type=type(obj)))


class StreamToLogger:
    def __init__(self, logger: logging.Logger, level: typing.Union[str, int], pass_through_stream: typing.TextIO = None) -> None:
        self.logger = logger
        self.level = logging._checkLevel(level)  # type: ignore
        self.pending_line = ""
        self.closed = False
        self.pass_through_stream = pass_through_stream

    # Here we are trying to test for the case of a recursive loop which can happen
    # if you are using "logging.StreamHandler" in your logging configuration (e.g., to
    # output logging to a console) and configure it after "redirect_to_logging' context
    # manager has been entered.
    def _check_recursion(self) -> bool:
        # We start at "2" so that we start from outside of this file.
        frame = sys._getframe(2)
        line_number = None
        try:
            i = 0
            # If loop is happening, it is generally looping inside less than 10 frames,
            # so we exit after 20 frames (just to make sure, all these values are ballpark
            # values) to optimize.
            while frame and i < 20:
                if frame.f_code.co_filename == __file__:
                    # The first (in fact the last from call perspective) time we are
                    # in this file.
                    if line_number is None:
                        line_number = frame.f_lineno
                    # If we were in the same file and line already higher in the stack,
                    # we are in a recursive loop.
                    elif line_number == frame.f_lineno:
                        return True
                frame = frame.f_back
                i += 1
        finally:
            del frame

        return False

    def write(self, buffer: str) -> int:
        if self.closed:
            raise ValueError("Stream is closed.")

        if self._check_recursion():
            # We are being called by a logger in a recursive loop. Because this message has already been logged,
            # it is safe for us to just drop it to break a recursive loop.
            return 0

        # We only write complete lines to the logger. Any incomplete line will be saved to "pending_line", and flushed
        # if "flush" is called or the context manager is closed.
        bytes_written = 0
        lines = (self.pending_line + buffer).split('\n')
        # Since we split on "\n", the last string in the list of lines will be an empty string if the last character
        # in the buffer is a newline, which is what we want in this case as it resets the "pending_line" to empty.
        # Otherwise the last string in the list of lines are characters after the last "\n", which is again what we
        # want, setting the "pending_line" to characters not logged this time.
        self.pending_line = lines[-1]
        for line in lines[:-1]:
            # Whitespace lines should not be logged.
            if line.strip():
                self.logger.log(self.level, line.rstrip())
                bytes_written += len(line)

        if self.pass_through_stream is not None:
            self.pass_through_stream.write(buffer)

        return bytes_written

    def writelines(self, lines: typing.List[str]) -> None:
        if self.closed:
            raise ValueError("Stream is closed.")

        if self._check_recursion():
            # We are being called by a logger in a recursive loop. Because this message has already been logged,
            # it is safe for us to just drop it to break a recursive loop.
            return

        for line in lines:
            if line.strip():
                self.logger.log(self.level, line.rstrip())

        if self.pass_through_stream is not None:
            if hasattr(self.pass_through_stream, 'writelines'):
                self.pass_through_stream.writelines(lines)
            else:
                for line in lines:
                    self.pass_through_stream.write(line)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("Stream is closed.")

        if self.pending_line.strip():
            self.logger.log(self.level, self.pending_line.rstrip())

        if self.pass_through_stream is not None:
            self.pass_through_stream.flush()

    def close(self) -> None:
        if self.closed:
            return

        if self.pending_line.strip():
            self.logger.log(self.level, self.pending_line.rstrip())
        self.closed = True

    def seekable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        raise OSError("Stream is not seekable.")

    def tell(self) -> int:
        raise OSError("Stream is not seekable.")

    def truncate(self, size: int = None) -> int:
        raise OSError("Stream is not seekable.")

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def read(self, n: int = -1) -> typing.AnyStr:
        raise OSError("Stream is write-only.")

    def readline(self, limit: int = -1) -> typing.AnyStr:
        raise OSError("Stream is write-only.")

    def readlines(self, hint: int = -1) -> typing.List[typing.AnyStr]:
        raise OSError("Stream is write-only.")

    def fileno(self) -> int:
        raise OSError("Stream does not use a file descriptor.")


class redirect_to_logging(contextlib.AbstractContextManager):
    """
    A Python context manager which redirects all writes to stdout and stderr
    to Python logging.

    Primitives should use logging to log messages, but maybe they are not doing
    that or there are other libraries they are using which are not doing that.
    One can then use this context manager to assure that (at least all Python)
    writes to stdout and stderr by primitives are redirected to logging::

        with redirect_to_logging(logger=PrimitiveClass.logger):
            primitive = PrimitiveClass(...)
            primitive.set_training_data(...)
            primitive.fit(...)
            primitive.produce(...)
    """

    # These are class variables to ensure that they are shared among all instances.
    # We use a list to make this context manager re-entrant.
    _old_stdouts: typing.List[typing.TextIO] = []
    _old_stderrs: typing.List[typing.TextIO] = []

    def __init__(self, logger: logging.Logger = None, stdout_level: typing.Union[int, str] = 'INFO', stderr_level: typing.Union[int, str] = 'ERROR', pass_through: bool = True) -> None:
        if logger is None:
            self.logger = logging.getLogger('redirect')
        else:
            self.logger = logger

        self.stdout_level = logging._checkLevel(stdout_level)  # type: ignore
        self.stderr_level = logging._checkLevel(stderr_level)  # type: ignore
        self.pass_through = pass_through

    def __enter__(self) -> logging.Logger:
        self._old_stdouts.append(sys.stdout)
        self._old_stderrs.append(sys.stderr)
        if self.pass_through:
            stdout_pass_through = self._old_stdouts[0]
            stderr_pass_through = self._old_stderrs[0]
        else:
            stdout_pass_through = None
            stderr_pass_through = None
        sys.stdout = typing.cast(typing.TextIO, StreamToLogger(self.logger, self.stdout_level, stdout_pass_through))
        sys.stderr = typing.cast(typing.TextIO, StreamToLogger(self.logger, self.stdout_level, stderr_pass_through))
        return self.logger

    def __exit__(self, exc_type: typing.Optional[typing.Type[BaseException]],
                 exc_value: typing.Optional[BaseException],
                 traceback: typing.Optional[types.TracebackType]) -> typing.Optional[bool]:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._old_stdouts.pop()
        sys.stderr = self._old_stderrs.pop()
        return None


class CallbackHandler(logging.Handler):
    """
    Calls a ``callback`` with logging records as they are without any conversion except for:

     * formatting the logging message and adding it to the record object
     * assuring ``asctime`` is set
     * converts exception ``exc_info`` into exception's name
     * making sure ``args`` are JSON-compatible or removing it
     * making sure there are no null values
    """

    def __init__(self, callback: typing.Callable) -> None:
        super().__init__(logging.DEBUG)

        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(self.prepare(record))
        except Exception:
            self.handleError(record)

    def prepare(self, record: logging.LogRecord) -> typing.Dict:
        self.format(record)

        # If "asctime" is not set, we do it ourselves.
        if not hasattr(record, 'asctime'):
            if self.formatter:
                fmt = self.formatter
            else:
                fmt = logging._defaultFormatter  # type: ignore
            record.asctime = fmt.formatTime(record, fmt.datefmt)

        output = copy.copy(record.__dict__)

        # Exceptions are not JSON compatible.
        if 'exc_info' in output:
            if output['exc_info']:
                if isinstance(output['exc_info'], BaseException):
                    output['exc_type'] = type_to_str(type(output['exc_info']))
                else:
                    output['exc_type'] = type_to_str(type(output['exc_info'][1]))
            del output['exc_info']

        if 'args' in output:
            try:
                output['args'] = to_json_structure(output['args'])
            except Exception:
                # We assume this means "args" is not JSON compatible.
                del output['args']

        # We iterate over a list so that we can change dict while iterating.
        for key, value in list(output.items()):
            if value is None:
                del output[key]

        return output


def _called_from_outside(modules: typing.Sequence[types.ModuleType]) -> bool:
    # 0 == this function, 1 == wrapper, 2 == caller
    frame = sys._getframe(2)
    try:
        if not frame:
            caller_module_name = None
        else:
            caller_module_name = frame.f_globals.get('__name__', None)
    finally:
        del frame

    return all(caller_module_name != module.__name__ for module in modules)


def _decorate_all_methods(modules: typing.Sequence[types.ModuleType], src_obj: typing.Any, dst_obj: typing.Any, decorator: typing.Callable, ignore: typing.Set) -> None:
    for name, function in inspect.getmembers(src_obj):
        if name.startswith('_'):
            continue

        if name in ignore:
            continue

        # Wrap the method with the decorator.
        if isinstance(function, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType)):
            # For simplicity we use the name of the first module.
            decorated_function = decorator(modules, modules[0].__name__, name, function)
            setattr(dst_obj, name, decorated_function)

            # When functions are imported to other modules, we have to update those imported functions as well.
            # Here we iterate over known modules and check if original function was copied over. If it was,
            # we set it to the new decorated function.
            for module in modules:
                if getattr(module, name, None) == function:
                    setattr(module, name, decorated_function)


_random_warnings_enabled: typing.List[bool] = []
_random_sources_patched = False


def _random_warning_decorator(modules: typing.Sequence[types.ModuleType], module_path: str, function_name: str, f: typing.Callable) -> typing.Callable:
    @functools.wraps(f)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        global _random_warnings_enabled

        # Some methods call into other methods. We do not want to issue a warning in such cases.
        if _random_warnings_enabled and _random_warnings_enabled[-1] and _called_from_outside(modules):
            log_once(
                logger,
                logging.WARNING,
                "Using global/shared random source using '%(module_path)s.%(function_name)s' can make execution not reproducible.",
                {
                    'module_path': module_path,
                    'function_name': function_name,
                },
                stack_info=True,
            )

        return f(*args, **kwargs)

    return wrapper


class _RandomState(numpy_random.RandomState):
    """
    A subclass just so that we can set somewhere decorated methods. The original class is read-only.
    """


def _patch_random_sources() -> None:
    global _random_sources_patched

    if _random_sources_patched:
        return
    _random_sources_patched = True

    # We patch the global Python random number generator instance by decorating all methods.
    # Used to support "global_randomness_warning" context manager.
    # We do not issue warning for calling "getstate".
    _decorate_all_methods([random], random._inst, random._inst, _random_warning_decorator, {'getstate'})  # type: ignore

    # For global NumPy random number generator we create a new random state instance first (of our subclass),
    # and copy the state over. This is necessary because original random state instance has read-only methods.
    old_rand = numpy.random.mtrand._rand
    numpy.random.mtrand._rand = _RandomState()
    numpy.random.mtrand._rand.set_state(old_rand.get_state())

    # We do not issue warning for calling "get_state".
    _decorate_all_methods([numpy.random, numpy.random.mtrand], old_rand, numpy.random.mtrand._rand, _random_warning_decorator, {'get_state'})  # type: ignore

    if hasattr(numpy_random, 'default_rng'):
        old_default_rng = numpy_random.default_rng

        def default_rng(seed: typing.Any = None) -> typing.Any:
            if seed is None:
                log_once(
                    logger,
                    logging.WARNING,
                    "Using 'numpy.random.default_rng' without a seed can make execution not reproducible.",
                    stack_info=True,
                )

            return old_default_rng(seed)

        numpy_random.default_rng = default_rng


class global_randomness_warning(contextlib.AbstractContextManager):
    """
    A Python context manager which issues a warning if global sources of
    randomness are used. Currently it checks Python built-in global random
    source, NumPy global random source, and NumPy ``default_rng`` being
    used without a seed.
    """

    def __init__(self, enable: bool = True) -> None:
        self.enable = enable
        _patch_random_sources()

    def __enter__(self) -> None:
        _random_warnings_enabled.append(self.enable)

    def __exit__(self, exc_type: typing.Optional[typing.Type[BaseException]],
                 exc_value: typing.Optional[BaseException],
                 traceback: typing.Optional[types.TracebackType]) -> typing.Optional[bool]:
        _random_warnings_enabled.pop()
        return None


def get_full_name(value: typing.Any) -> str:
    return '{module}.{name}'.format(module=value.__module__, name=value.__name__)


def has_duplicates(data: typing.Sequence) -> bool:
    """
    Returns ``True`` if ``data`` has duplicate elements.

    It works both with hashable and not-hashable elements.
    """

    try:
        return len(set(data)) != len(data)
    except TypeError:
        n = len(data)
        for i in range(n):
            for j in range(i + 1, n):
                if data[i] == data[j]:
                    return True
        return False


@contextlib.contextmanager
def silence() -> typing.Generator:
    """
    Hides logging and stdout output.
    """

    with unittest.TestCase().assertLogs(level=logging.DEBUG):
        with redirect_to_logging(pass_through=False):
            # Just to log something, otherwise "assertLogs" can fail.
            logging.getLogger().debug("Silence.")

            yield


@deprecate.arguments('source', message="argument ignored")
def columns_sum(inputs: typing.Any, *, source: typing.Any = None) -> typing.Any:
    """
    Computes sum per column.
    """

    # Importing here to prevent import cycle.
    from d3m import container

    if isinstance(inputs, container.DataFrame):  # type: ignore
        results = container.DataFrame(inputs.agg(['sum']).reset_index(drop=True), generate_metadata=True)  # type: ignore
        return results

    elif isinstance(inputs, container.ndarray) and len(inputs.shape) == 2:
        return numpy.sum(inputs, axis=0, keepdims=True)

    else:
        raise exceptions.InvalidArgumentTypeError("Unsupported container type to sum: {type}".format(
            type=type(inputs),
        ))


def list_files(base_directory: str) -> typing.Sequence[str]:
    files = []

    base_directory = base_directory.rstrip(os.path.sep)
    base_directory_prefix_length = len(base_directory) + 1
    for dirpath, dirnames, filenames in os.walk(base_directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)

            # We do not use "os.path.relpath" because it is to general
            # and it first try to construct absolute path which is slow.
            files.append(filepath[base_directory_prefix_length:])

    # We sort to have a canonical order.
    files = sorted(files)

    return files


def _is_int(typ: type) -> bool:
    # We support more types than those listed in "d3m.types.simple_data_types".
    return issubclass(typ, (int, numpy.integer, numbers.Integral))


def is_int(typ: type) -> bool:
    return _is_int(typ) and not issubclass(typ, bool)


def _is_float(typ: type) -> bool:
    # We support more types than those listed in "d3m.types.simple_data_types".
    return issubclass(typ, (float, numpy.float32, numpy.float64, decimal.Decimal, numbers.Real))


def is_float(typ: type) -> bool:
    return _is_float(typ) and not is_int(typ)


def is_numeric(typ: type) -> bool:
    return is_int(typ) or _is_float(typ)


def compute_hash_id(obj: typing.Dict) -> str:
    """
    Input should be a JSON compatible structure.
    """

    obj = copy.copy(obj)

    if 'id' in obj:
        del obj['id']

    # We iterate over a list so that we can change dict while iterating.
    for key in list(obj.keys()):
        # Do not count any private field into hash.
        if key.startswith('_'):
            del obj[key]

    # We have to use "normalize_numbers" first so that we normalize numbers.
    # We cannot do this just with a custom encoder because encoders are not
    # called for float values so we cannot handle them there.
    # See: https://bugs.python.org/issue36841
    to_hash_id = json.dumps(normalize_numbers(obj), sort_keys=True)

    return str(uuid.uuid5(HASH_ID_NAMESPACE, to_hash_id))


def compute_digest(obj: typing.Dict, extra_data: bytes = None) -> str:
    """
    Input should be a JSON compatible structure.
    """

    obj = copy.copy(obj)

    if 'digest' in obj:
        del obj['digest']

    # We iterate over a list so that we can change dict while iterating.
    for key in list(obj.keys()):
        # Do not count any private field into digest.
        if key.startswith('_'):
            del obj[key]

    # We have to use "normalize_numbers" first so that we normalize numbers.
    # We cannot do this just with a custom encoder because encoders are not
    # called for float values so we cannot handle them there.
    # See: https://bugs.python.org/issue36841
    to_digest = json.dumps(normalize_numbers(obj), sort_keys=True)

    digest = hashlib.sha256(to_digest.encode('utf8'))

    if extra_data is not None:
        digest.update(extra_data)

    return digest.hexdigest()


def is_sequence(value: typing.Any) -> bool:
    return isinstance(value, typing.Sequence) and not isinstance(value, (str, bytes))


def get_dict_path(input_dict: typing.Dict, path: typing.Sequence[typing.Any]) -> typing.Any:
    value: typing.Any = input_dict

    for segment in path:
        value = value.get(segment, None)

        if value is None:
            return None

    return value


def set_dict_path(input_dict: typing.Dict, path: typing.Sequence[typing.Any], value: typing.Any) -> None:
    if not path:
        raise exceptions.InvalidArgumentValueError("\"path\" has to be non-empty.")

    for segment in path[:-1]:
        if segment not in input_dict:
            input_dict[segment] = {}
        input_dict = input_dict[segment]

    input_dict[path[-1]] = value


def register_yaml_representers() -> None:
    def yaml_representer_numpy_float(dumper: yaml.Dumper, data: typing.Any) -> typing.Any:
        return dumper.represent_float(float(data))

    def yaml_representer_numpy_int(dumper: yaml.Dumper, data: typing.Any) -> typing.Any:
        return dumper.represent_int(int(data))

    def yaml_representer_numpy_bool(dumper: yaml.Dumper, data: typing.Any) -> typing.Any:
        return dumper.represent_bool(bool(data))

    representers = [
        {'type': numpy.float32, 'representer': yaml_representer_numpy_float},
        {'type': numpy.float64, 'representer': yaml_representer_numpy_float},
        {'type': numpy.int32, 'representer': yaml_representer_numpy_int},
        {'type': numpy.int64, 'representer': yaml_representer_numpy_int},
        {'type': numpy.integer, 'representer': yaml_representer_numpy_int},
        {'type': numpy.bool_, 'representer': yaml_representer_numpy_bool},
    ]

    for representer in representers:
        yaml_add_representer(representer['type'], representer['representer'])


# Registers additional regexp for floating point resolver.
# See: https://github.com/yaml/pyyaml/issues/173
def register_yaml_resolvers() -> None:
    tag = 'tag:yaml.org,2002:float'
    regexp = re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$''', re.X)
    first = list(u'-+0123456789.')

    yaml.Dumper.add_implicit_resolver(tag, regexp, first)
    yaml.SafeDumper.add_implicit_resolver(tag, regexp, first)
    yaml.Loader.add_implicit_resolver(tag, regexp, first)
    yaml.SafeLoader.add_implicit_resolver(tag, regexp, first)

    if yaml.__with_libyaml__:
        yaml.CDumper.add_implicit_resolver(tag, regexp, first)  # type: ignore
        yaml.CSafeDumper.add_implicit_resolver(tag, regexp, first)  # type: ignore
        yaml.CLoader.add_implicit_resolver(tag, regexp, first)  # type: ignore
        yaml.CSafeLoader.add_implicit_resolver(tag, regexp, first)  # type: ignore


def matches_structural_type(source_structural_type: type, target_structural_type: typing.Union[str, type]) -> bool:
    if isinstance(target_structural_type, str):
        return type_to_str(source_structural_type) == target_structural_type
    else:
        return is_subclass(source_structural_type, target_structural_type)


# Register YAML representers and resolvers.
register_yaml_representers()
register_yaml_resolvers()


class PMap(pyrsistent.PMap):
    """
    Extends `pyrsistent.PMap` to (by default) iterate over its items in sorted order.
    """

    def iterkeys(self, *, sort: bool = True, reverse: bool = False) -> typing.Iterable:
        for k, _ in self.iteritems(sort=sort, reverse=reverse):
            yield k

    def itervalues(self, *, sort: bool = True, reverse: bool = False) -> typing.Iterable:
        for _, v in self.iteritems(sort=sort, reverse=reverse):
            yield v

    def iteritems(self, *, sort: bool = True, reverse: bool = False) -> typing.Iterable:
        if sort:
            yield from sorted(super().iteritems(), key=operator.itemgetter(0), reverse=reverse)
        else:
            yield from super().iteritems()

    # In Python 3 this is also an iterable.
    def values(self, *, sort: bool = True, reverse: bool = False) -> typing.Iterable:
        return self.itervalues(sort=sort, reverse=reverse)

    # In Python 3 this is also an iterable.
    def keys(self, *, sort: bool = True, reverse: bool = False) -> typing.Iterable:
        return self.iterkeys(sort=sort, reverse=reverse)

    # In Python 3 this is also an iterable.
    def items(self, *, sort: bool = True, reverse: bool = False) -> typing.Iterable:
        return self.iteritems(sort=sort, reverse=reverse)

    def evolver(self) -> 'Evolver':
        return Evolver(self)

    def __reduce__(self) -> typing.Tuple[typing.Callable, typing.Tuple[typing.Dict]]:
        return pmap, (dict(self),)


class Evolver(pyrsistent.PMap._Evolver):
    def persistent(self) -> PMap:
        if self.is_dirty():
            self._original_pmap = PMap(self._size, self._buckets_evolver.persistent())

        return self._original_pmap


# It is OK to use a mutable default value here because it is never changed in-place.
def pmap(initial: typing.Mapping = {}, pre_size: int = 0) -> PMap:
    super_pmap = pyrsistent.pmap(initial, pre_size)

    return PMap(super_pmap._size, super_pmap._buckets)


EMPTY_PMAP = pmap()


def is_uri(uri: str) -> bool:
    """
    Test if a given string is an URI.

    Parameters
    ----------
    uri:
        A potential URI to test.

    Returns
    -------
    ``True`` if string is an URI, ``False`` otherwise.
    """

    try:
        parsed_uri = url_parse.urlparse(uri, allow_fragments=False)
    except Exception:
        return False

    return parsed_uri.scheme != ''


def fix_uri(uri: str, *, allow_relative_path: bool = True) -> str:
    """
    Make a real file URI from a path.

    Parameters
    ----------
    uri:
        An input URI.
    allow_relative_path:
        Allow path to be relative?

    Returns
    -------
    A fixed URI.
    """

    if is_uri(uri):
        return uri

    if not uri.startswith('/') and not allow_relative_path:
        raise exceptions.InvalidArgumentValueError(f"Path cannot be relative: {uri}")

    # Make absolute and normalize at the same time.
    uri = os.path.abspath(uri)

    return 'file://{uri}'.format(uri=uri)


def outside_package_context() -> typing.Optional[deprecate.Context]:
    frame = sys._getframe(1)
    try:
        while frame:
            if frame.f_code.co_filename == '<stdin>' or os.path.commonpath([PACKAGE_BASE, frame.f_code.co_filename]) != PACKAGE_BASE:
                return deprecate.Context(None, None, frame.f_code.co_filename, frame.f_globals.get('__name__', None), frame.f_lineno)

            frame = frame.f_back

    finally:
        del frame

    return None


already_logged: typing.Set[typing.Tuple[deprecate.Context, deprecate.Context]] = set()


def log_once(logger: logging.Logger, level: int, msg: str, *args: typing.Any, **kwargs: typing.Any) -> None:
    frame = sys._getframe(1)
    try:
        if not frame:
            function_context = None
        else:
            function_context = deprecate.Context(str(level), msg, frame.f_code.co_filename, frame.f_globals.get('__name__', None), frame.f_lineno)
    finally:
        del frame

    module_context = outside_package_context()

    context = (module_context, function_context)

    if context in already_logged:
        return

    if module_context is not None and function_context is not None:
        already_logged.add(context)

    logger.log(level, msg, *args, **kwargs)


# A workaround to handle also binary stdin/stdout.
# See: https://gitlab.com/datadrivendiscovery/d3m/issues/353
# See: https://bugs.python.org/issue14156
# Moreover, if filename ends in ".gz" it decompresses the file as well.
class FileType(argparse.FileType):
    def __call__(self, string: str) -> typing.IO[typing.Any]:
        if string.endswith('.gz'):
            # "gzip.open" has as a default binary mode,
            # but we want text mode as a default.
            if 't' not in self._mode and 'b' not in self._mode:  # type: ignore
                mode = self._mode + 't'  # type: ignore
            else:
                mode = self._mode  # type: ignore

            try:
                return gzip.open(string, mode=mode, encoding=self._encoding, errors=self._errors)  # type: ignore
            except OSError as error:
                message = argparse._("can't open '%s': %s")  # type: ignore
                raise argparse.ArgumentTypeError(message % (string, error))

        handle = super().__call__(string)

        if string == '-' and 'b' in self._mode:  # type: ignore
            handle = handle.buffer  # type: ignore

        return handle


def open(file: str, mode: str = 'r', buffering: int = -1, encoding: str = None, errors: str = None) -> typing.IO[typing.Any]:
    try:
        return FileType(mode=mode, bufsize=buffering, encoding=encoding, errors=errors)(file)
    except argparse.ArgumentTypeError as error:
        original_error = error.__context__

    # So that we are outside of the except clause.
    raise original_error


def filter_local_location_uris(doc: typing.Dict, *, empty_value: typing.Any = None) -> None:
    if 'location_uris' in doc:
        location_uris = []
        for location_uri in doc['location_uris']:
            try:
                parsed_uri = url_parse.urlparse(location_uri, allow_fragments=False)
            except Exception:
                continue

            if parsed_uri.scheme == 'file':
                continue

            location_uris.append(location_uri)

        if location_uris:
            doc['location_uris'] = location_uris
        elif empty_value is not None:
            doc['location_uris'] = empty_value
        else:
            del doc['location_uris']

    if 'location_base_uris' in doc:
        location_base_uris = []
        for location_base_uri in doc['location_base_uris']:
            try:
                parsed_uri = url_parse.urlparse(location_base_uri, allow_fragments=False)
            except Exception:
                continue

            if parsed_uri.scheme == 'file':
                continue

            location_base_uris.append(location_base_uri)

        if location_base_uris:
            doc['location_base_uris'] = location_base_uris
        elif empty_value is not None:
            doc['location_base_uris'] = empty_value
        else:
            del doc['location_base_uris']


def json_structure_equals(
    obj1: typing.Any, obj2: typing.Any, ignore_keys: typing.Set = None,
) -> bool:
    """
    Parameters
    ----------
    obj1:
        JSON serializable object to compare with ``obj2``.
    obj2:
        JSON serializable object to compare with ``obj1``.
    ignore_keys:
        If ``obj1`` and ``obj2`` are of type ``Mapping``, any keys found in this set will not be considered to
        determine whether ``obj1`` and ``obj2`` are equal.

    Returns
    -------
    A boolean indicating whether ``obj1`` and ``obj2`` are equal.
    """

    if ignore_keys is None:
        ignore_keys = set()

    if isinstance(obj1, collections.Mapping) and isinstance(obj2, collections.Mapping):
        for key1 in obj1:
            if key1 in ignore_keys:
                continue
            if key1 not in obj2:
                return False
            if not json_structure_equals(obj1[key1], obj2[key1], ignore_keys):
                return False

        for key2 in obj2:
            if key2 in ignore_keys:
                continue
            if key2 not in obj1:
                return False
            # Already checked if values are equal.

        return True

    elif is_sequence(obj1) and is_sequence(obj2):
        if len(obj1) != len(obj2):
            return False
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            if not json_structure_equals(item1, item2, ignore_keys):
                return False
        return True

    else:
        return obj1 == obj2


@functools.lru_cache()
def get_datasets_and_problems(
    datasets_dir: str, handle_score_split: bool = True,
) -> typing.Tuple[typing.Dict[str, str], typing.Dict[str, str]]:
    if datasets_dir is None:
        raise exceptions.InvalidArgumentValueError("Datasets directory has to be provided.")

    datasets: typing.Dict[str, str] = {}
    problem_descriptions: typing.Dict[str, str] = {}
    problem_description_contents: typing.Dict[str, typing.Dict] = {}

    for dirpath, dirnames, filenames in os.walk(datasets_dir, followlinks=True):
        if 'datasetDoc.json' in filenames:
            # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
            # exists in raw data filename).
            dirnames[:] = []

            dataset_path = os.path.join(os.path.abspath(dirpath), 'datasetDoc.json')

            try:
                with open(dataset_path, 'r', encoding='utf8') as dataset_file:
                    dataset_doc = json.load(dataset_file)

                dataset_id = dataset_doc['about']['datasetID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding dataset ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if handle_score_split and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and dataset_id.endswith('_TEST'):
                    dataset_id = dataset_id[:-5] + '_SCORE'

                if dataset_id in datasets:
                    logger.warning(
                        "Duplicate dataset ID '%(dataset_id)s': '%(old_dataset)s' and '%(dataset)s'", {
                            'dataset_id': dataset_id,
                            'dataset': dataset_path,
                            'old_dataset': datasets[dataset_id],
                        },
                    )
                else:
                    datasets[dataset_id] = dataset_path

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read dataset '%(dataset)s'.", {
                        'dataset': dataset_path,
                    },
                )

        if 'problemDoc.json' in filenames:
            # We continue traversing further in this case.

            problem_path = os.path.join(os.path.abspath(dirpath), 'problemDoc.json')

            try:
                with open(problem_path, 'r', encoding='utf8') as problem_file:
                    problem_doc = json.load(problem_file)

                problem_id = problem_doc['about']['problemID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding problem ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if handle_score_split and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and problem_id.endswith('_TEST'):
                    problem_id = problem_id[:-5] + '_SCORE'

                    # Also update dataset references.
                    for data in problem_doc.get('inputs', {}).get('data', []):
                        if data['datasetID'].endswith('_TEST'):
                            data['datasetID'] = data['datasetID'][:-5] + '_SCORE'

                with open(problem_path, 'r', encoding='utf8') as problem_file:
                    problem_description = json.load(problem_file)

                if problem_id in problem_descriptions and problem_description != problem_description_contents[problem_id]:
                    logger.warning(
                        "Duplicate problem ID '%(problem_id)s': '%(old_problem)s' and '%(problem)s'", {
                            'problem_id': problem_id,
                            'problem': problem_path,
                            'old_problem': problem_descriptions[problem_id],
                        },
                    )
                else:
                    problem_descriptions[problem_id] = problem_path
                    problem_description_contents[problem_id] = problem_description

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read problem description '%(problem)s'.", {
                        'problem': problem_path,
                    },
                )

    return datasets, problem_descriptions
