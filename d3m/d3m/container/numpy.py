import datetime
import typing

import numpy  # type: ignore

from d3m import deprecate
from d3m.metadata import base as metadata_base

# See: https://gitlab.com/datadrivendiscovery/d3m/issues/66
try:
    from pyarrow import lib as pyarrow_lib  # type: ignore
except ModuleNotFoundError:
    pyarrow_lib = None

__all__ = ('ndarray',)

# This implementation is based on these guidelines:
# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html

N = typing.TypeVar('N', bound='ndarray')


# TODO: We could implement also __array_ufunc__ and adapt metadata as well after in-place changes to data?
class ndarray(numpy.ndarray):
    """
    Extended `numpy.ndarray` with the ``metadata`` attribute.

    Parameters
    ----------
    input_array:
        Anything array-like to create an instance from. Including lists and standard numpy arrays.
    metadata:
        Optional initial metadata for the top-level of the array, or top-level metadata to be updated
        if ``input_array`` is another instance of this array class.
    generate_metadata:
        Automatically generate and update the metadata.
    check:
        DEPRECATED: argument ignored.
    source:
        DEPRECATED: argument ignored.
    timestamp:
        DEPRECATED: argument ignored.

    Attributes
    ----------
    metadata:
        Metadata associated with the array.
    """

    metadata: metadata_base.DataMetadata

    @deprecate.arguments('source', 'timestamp', 'check', message="argument ignored")
    def __new__(cls: typing.Type[N], input_array: typing.Sequence, metadata: typing.Dict[str, typing.Any] = None, *,
                generate_metadata: bool = False, check: bool = True, source: typing.Any = None, timestamp: datetime.datetime = None) -> N:
        array = numpy.asarray(input_array).view(cls)

        # Importing here to prevent import cycle.
        from d3m import types

        if isinstance(input_array, types.Container):
            if isinstance(input_array, ndarray):
                # We made a copy, so we do not have to generate metadata.
                array.metadata = input_array.metadata  # type: ignore
            else:
                array.metadata = input_array.metadata
                if generate_metadata:
                    array.metadata = array.metadata.generate(array)

            if metadata is not None:
                array.metadata = array.metadata.update((), metadata)
        else:
            array.metadata = metadata_base.DataMetadata(metadata)
            if generate_metadata:
                array.metadata = array.metadata.generate(array)

        return array

    def __array_finalize__(self, obj: typing.Any) -> None:
        # If metadata attribute already exists.
        if hasattr(self, 'metadata'):
            return

        if obj is not None and isinstance(obj, ndarray) and hasattr(obj, 'metadata'):
            # TODO: We could adapt (if this is after a slice) metadata instead of just copying?
            self.metadata: metadata_base.DataMetadata = obj.metadata
        else:
            self.metadata = metadata_base.DataMetadata()

    def __reduce__(self) -> typing.Tuple:
        reduced = list(super().__reduce__())

        reduced[2] = {
            'numpy': reduced[2],
            'metadata': self.metadata,
        }

        return tuple(reduced)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state['numpy'])

        self.metadata = state['metadata']


def ndarray_serializer(obj: ndarray) -> dict:
    data = {
        'metadata': obj.metadata,
        'numpy': obj.view(numpy.ndarray),
    }

    if type(obj) is not ndarray:
        data['type'] = type(obj)

    return data


def ndarray_deserializer(data: dict) -> ndarray:
    array = data['numpy'].view(data.get('type', ndarray))
    array.metadata = data['metadata']
    return array


if pyarrow_lib is not None:
    pyarrow_lib._default_serialization_context.register_type(
        ndarray, 'd3m.ndarray',
        custom_serializer=ndarray_serializer,
        custom_deserializer=ndarray_deserializer,
    )
