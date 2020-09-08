import importlib.abc
import importlib.machinery
import logging
import pkg_resources
import sys
import types
import typing

__all__ = ('setup',)

logger = logging.getLogger(__name__)

# For which entry points we already warned that they are ignored?
_ignored_entry_points: typing.Set[str] = set()


def entry_points() -> typing.Iterator[pkg_resources.EntryPoint]:
    """
    Makes sure that if two entry points are conflicting (one has a path
    pointing to a primitive, and another is a path pointing to a module containing
    other modules or primitives), the latter entry point is returned
    while the former is ignored (and warned about). This makes loading primitives
    deterministic.

    We iterate every time over entry points because maybe entry points have changed.
    """

    modules = set(tuple(entry_point.name.split('.')[:-1]) for entry_point in pkg_resources.iter_entry_points('d3m.primitives'))

    for entry_point in pkg_resources.iter_entry_points('d3m.primitives'):
        primitive_path = tuple(entry_point.name.split('.'))

        # "primitive_path" starts with a module path the last segment is a class name. If it exists
        # as a whole among what is seen as modules for all primitives, we have a conflict.
        if primitive_path in modules:
            if entry_point.name not in _ignored_entry_points:
                _ignored_entry_points.add(entry_point.name)
                logger.warning("An entry point for a primitive is conflicting with another entry point which has it as a module: %(entry_point_name)s", {'entry_point_name': entry_point.name})
        else:
            yield entry_point


class ModuleType(types.ModuleType):
    """
    A module which loads primitives on demand under ``d3m.primitives`` namespace.
    """

    def __dir__(self) -> typing.Sequence[str]:
        """
        Adds to listed attributes of a module all primitive classes known from
        entry points to be available under this module.

        They are not necessary loadable (trying to access them tries to load a primitive which
        might fail) and it is not yet necessary that they are really pointing to primitive classes,
        because this method does not try to load them yet to determine any of that.

        Already loaded primitives and imported submodules are provided by parent implementation
        of "__dir__" already because they are real attributes of this module.

        We add only classes. Submodules are added as real attributes once they are
        explicitly imported. This mimics how things work for regular modules in Python.
        """

        entries = set(super().__dir__())

        current_module = self.__name__.split('.')

        for entry_point in entry_points():
            entry_point_name = ['d3m', 'primitives'] + entry_point.name.split('.')

            # We assume the last segment is a class name, so we remove it.
            entry_point_module = entry_point_name[:-1]

            # If an entry point points to a class directly under this module, we add that class' name.
            if current_module == entry_point_module:
                # The last segment is a class name.
                entries.add(entry_point_name[-1])

        return list(entries)

    def __getattr__(self, item: str) -> typing.Any:
        """
        This method is called when there is no real attribute with name "item" already
        present in this module object (so not an existing method, an already loaded primitive,
        or already imported submodule).

        If it looks like "name" is pointing to a primitive, we load the primitive here and add
        it to the module object as a real attribute by calling "register_primitive".

        If it does not look like a primitive, we raise an exception and Python importing logic
        tries to import the module instead.
        """

        # Importing here to prevent import cycle.
        from d3m import index

        item_path = self.__name__.split('.') + [item]

        for entry_point in entry_points():
            entry_point_name = ['d3m', 'primitives'] + entry_point.name.split('.')

            # We assume for the last segment to be a class, so the full path has to match
            # for path to look like it is pointing to a primitive's class.
            if item_path == entry_point_name:
                primitive = None
                try:
                    logger.debug("Loading entry point '%(entry_point_name)s'.", {'entry_point_name': entry_point.name})
                    entry_point.require()
                    primitive = entry_point.resolve()  # type: ignore
                except pkg_resources.ResolutionError as error:
                    logger.warning("While loading primitive '%(entry_point_name)s', an error has been detected: %(error)s", {'entry_point_name': entry_point.name, 'error': error})
                    logger.warning("Attempting to load primitive '%(entry_point_name)s' without checking requirements.", {'entry_point_name': entry_point.name})

                # There was an error, so we try again without checking requirements.
                if primitive is None:
                    primitive = entry_point.resolve()  # type: ignore

                try:
                    # We set the sentinel so that when during registration attribute with name "name"
                    # is accessed this method is not called again (because a real attribute already
                    # exists) but the sentinel is returned.
                    setattr(self, item, index._SENTINEL)
                    index.register_primitive('.'.join(entry_point_name), primitive)
                except Exception:
                    if getattr(self, item) is index._SENTINEL:
                        delattr(self, item)
                    raise

                # Calling "register_primitive" should set a real attribute on this module object.
                assert getattr(self, item) is primitive

                return primitive

        raise AttributeError('module \'{name}\' has no attribute \'{item}\''.format(name=self.__name__, item=item))


class Loader(importlib.abc.Loader):
    """
    A loader which returns modules of our subclass.
    """

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        return ModuleType(spec.name, ModuleType.__doc__)

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class MetaPathFinder(importlib.abc.MetaPathFinder):
    """
    A finder for ``d3m.primitives`` namespace which uses our loader for entries in entry points.
    """

    def find_spec(self, fullname, path, target=None):  # type: ignore
        if not fullname.startswith('d3m.primitives'):
            return None

        if fullname == 'd3m.primitives':
            return importlib.machinery.ModuleSpec(fullname, Loader(), is_package=True)

        name = fullname.split('.')

        for entry_point in entry_points():
            entry_point_name = ['d3m', 'primitives'] + entry_point.name.split('.')

            # We assume the last segment is a class name, so we remove it.
            entry_point_module = entry_point_name[:-1]

            # There is at least one entry point having this name as its module,
            # so we return a module.
            if len(entry_point_module) >= len(name) and entry_point_module[0:len(name)] == name:
                return importlib.machinery.ModuleSpec(fullname, Loader(), is_package=True)

        return None


def setup() -> None:
    """
    Expose all primitives under the same ``d3m.primitives`` namespace.

    This is achieved using Python entry points. Python packages containing primitives
    can register them and expose them under the common namespace by adding an entry
    like the following to package's ``setup.py``::

        entry_points = {
            'd3m.primitives': [
                'primitive_namespace.PrimitiveName = my_package.my_module:PrimitiveClassName',
            ],
        },

    The example above would expose the ``my_package.my_module.PrimitiveClassName`` primitive under
    ``d3m.primitives.primitive_namespace.PrimitiveName``.
    """

    sys.meta_path.append(MetaPathFinder())
