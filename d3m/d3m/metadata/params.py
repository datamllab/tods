import typing

from d3m import exceptions, utils


class ParamsMeta(utils.AbstractMetaclass):
    def __new__(mcls, class_name, bases, namespace, **kwargs):  # type: ignore
        for name, value in namespace.items():
            if name.startswith('_'):
                continue

            if utils.is_class_method_on_class(value) or utils.is_instance_method_on_class(value):
                continue

            raise TypeError("Only methods and attribute type annotations can be defined on Params class, not '{name}'.".format(name=name))

        class_params_items = {}
        class_annotations = namespace.get('__annotations__', {})

        for name, value in class_annotations.items():
            value = typing._type_check(value, "Each annotation must be a type.")

            if name in namespace:
                # Just update the annotation.
                class_annotations[name] = value
            else:
                # Extract annotation out.
                class_params_items[name] = value

        for name in class_params_items.keys():
            del class_annotations[name]

        # Set back updated annotations.
        namespace['__annotations__'] = class_annotations

        params_items = {}

        for base in reversed(bases):
            params_items.update(base.__dict__.get('__params_items__', {}))

        params_items.update(class_params_items)

        namespace['__params_items__'] = params_items

        return super().__new__(mcls, class_name, bases, namespace, **kwargs)


class Params(dict, metaclass=ParamsMeta):
    """
    A base class to be subclassed and used as a type for ``Params`` type
    argument in primitive interfaces. An instance of this subclass should
    be returned from primitive's ``get_params`` method, and accepted in
    ``set_params``.

    You should subclass the class and set type annotations on class attributes
    for params available in the class.

    When creating an instance of the class, all parameters have to be provided.
    """

    def __init__(self, other: typing.Dict[str, typing.Any] = None, **values: typing.Any) -> None:
        if other is None:
            other = {}

        values = dict(other, **values)

        params_keys = set(self.__params_items__.keys())  # type: ignore
        values_keys = set(values.keys())

        missing = params_keys - values_keys
        if len(missing):
            raise exceptions.InvalidArgumentValueError("Not all parameters are specified: {missing}".format(missing=missing))

        extra = values_keys - params_keys
        if len(extra):
            raise exceptions.InvalidArgumentValueError("Additional parameters are specified: {extra}".format(extra=extra))

        for name, value in values.items():
            value_type = self.__params_items__[name]  # type: ignore
            if not utils.is_instance(value, value_type):
                raise exceptions.InvalidArgumentTypeError("Value '{value}' for parameter '{name}' is not an instance of the type: {value_type}".format(value=value, name=name, value_type=value_type))

        super().__init__(values)

    def __setitem__(self, key, value):  # type: ignore
        if key not in self.__params_items__:
            raise ValueError("Additional parameter is specified: {key}".format(key=key))

        value_type = self.__params_items__[key]
        if not utils.is_instance(value, value_type):
            raise TypeError("Value '{value}' for parameter '{name}' is not an instance of the type: {value_type}".format(value=value, name=key, value_type=value_type))

        return super().__setitem__(key, value)

    def __delitem__(self, key):  # type: ignore
        raise AttributeError("You cannot delete parameters.")

    def clear(self):  # type: ignore
        raise AttributeError("You cannot delete parameters.")

    def pop(self, key, default=None):  # type: ignore
        raise AttributeError("You cannot delete parameters.")

    def popitem(self):  # type: ignore
        raise AttributeError("You cannot delete parameters.")

    def setdefault(self, key, default=None):  # type: ignore
        if key not in self.__params_items__:
            raise ValueError("Additional parameter is specified: {key}".format(key=key))

        default_type = self.__params_items__[key]
        if not utils.is_instance(default, default_type):
            raise TypeError("Value '{value}' for parameter '{name}' is not an instance of the type: {value_type}".format(value=default, name=key, value_type=default_type))

        return super().setdefault(key, default)

    def update(self, other: typing.Dict[str, typing.Any] = None, **values: typing.Any) -> None:  # type: ignore
        if other is None:
            other = {}

        values = dict(other, **values)

        params_keys = set(self.__params_items__.keys())  # type: ignore
        values_keys = set(values.keys())

        extra = values_keys - params_keys
        if len(extra):
            raise ValueError("Additional parameters are specified: {extra}".format(extra=extra))

        for name, value in values.items():
            value_type = self.__params_items__[name]  # type: ignore
            if not utils.is_instance(value, value_type):
                raise TypeError("Value '{value}' for parameter '{name}' is not an instance of the type: {value_type}".format(value=value, name=name, value_type=value_type))

        super().update(values)

    def __repr__(self) -> str:
        return '{class_name}({super})'.format(class_name=type(self).__name__, super=super().__repr__())
