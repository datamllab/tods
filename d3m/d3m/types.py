import numpy  # type: ignore

from d3m import container

__all__ = ('Data', 'Container')

# Open an issue if these standard types are too restrictive for you,
# but the idea is that callers should know in advance which data types
# are being passed in and out of primitives to be able to implement
# their introspection, serialization, and so on.

simple_data_types = (
    str, bytes, bool, float, int, numpy.integer, numpy.float64, numpy.bool_, type(None),
)

# A tuple representing all standard container types.
Container = (
    container.ndarray, container.DataFrame,
    container.List, container.Dataset,
)

# A tuple representing all standard data types. Data types are those which
# can be contained inside container types.
Data = Container + simple_data_types + (dict,)
