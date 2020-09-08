Primitives discovery
================================

Primitives D3M namespace
------------------------

The :mod:`d3m.primitives` module exposes all primitives under the same
``d3m.primitives`` namespace.

This is achieved using :ref:`Python entry points <setuptools:Entry Points>`.
Python packages containing primitives should register them and expose
them under the common namespace by adding an entry like the following to
package's ``setup.py``:

.. code:: python

    entry_points = {
        'd3m.primitives': [
            'primitive_namespace.PrimitiveName = my_package.my_module:PrimitiveClassName',
        ],
    },

The example above would expose the
``my_package.my_module.PrimitiveClassName`` primitive under
``d3m.primitives.primitive_namespace.PrimitiveName``.

Configuring ``entry_points`` in your ``setup.py`` does not just put
primitives into a common namespace, but also helps with discovery of
your primitives on the system. Then your package with primitives just
have to be installed on the system and can be automatically discovered
and used by any other Python code.

    **Note:** Only primitive classes are available through the
    ``d3m.primitives`` namespace, no other symbols from a source
    module. In the example above, only ``PrimitiveClassName`` is
    available, not other symbols inside ``my_module`` (except if they
    are other classes also added to entry points).

    **Note:** Modules under ``d3m.primitives`` are created dynamically
    at run-time based on information from entry points. So some tools
    (IDEs, code inspectors, etc.) might not find them because there are
    no corresponding files and directories under ``d3m.primitives``
    module. You have to execute Python code for modules to be available.
    Static analysis cannot find them.

Primitives discovery on PyPi
----------------------------

To facilitate automatic discovery of primitives on PyPi (or any other
compatible Python Package Index), publish a package with a keyword
``d3m_primitive`` in its ``setup.py`` configuration:

.. code:: python

    keywords='d3m_primitive'

    **Note:** Be careful when automatically discovering, installing, and
    using primitives from unknown sources. While primitives are designed
    to be bootstrapable and automatically installable without human
    involvement, there are no isolation mechanisms yet in place for
    running potentially malicious primitives. Currently recommended way
    is to use manually curated lists of known primitives.

d3m.index API
--------------------------

The :mod:`d3m.index` module exposes the following Python utility functions.

``search``
~~~~~~~~~~

Returns a list of primitive paths (Python paths under ``d3m.primitives``
namespace) for all known (discoverable through entry points) primitives,
or limited by the ``primitive_path_prefix`` search argument.

``get_primitive``
~~~~~~~~~~~~~~~~~

Loads (if not already) a primitive class and returns it.

``get_primitive_by_id``
~~~~~~~~~~~~~~~~~~~~~~~

Returns a primitive class based on its ID from all currently loaded
primitives.

``get_loaded_primitives``
~~~~~~~~~~~~~~~~~~~~~~~~~

Returns a list of all currently loaded primitives.

``load_all``
~~~~~~~~~~~~

Loads all primitives available and populates ``d3m.primitives``
namespace with them.

``register_primitive``
~~~~~~~~~~~~~~~~~~~~~~

Registers a primitive under ``d3m.primitives`` namespace.

This is useful to register primitives not necessary installed on the
system or which are generated at runtime. It is also useful for testing
purposes.

``discover``
~~~~~~~~~~~~

Returns package names from PyPi which provide D3M primitives.

This is determined by them having a ``d3m_primitive`` among package
keywords.

Command line
------------

The :mod:`d3m.index` module also provides a command line interface by
running ``python3 -m d3m index``. The following commands are currently
available.

Use ``-h`` or ``--help`` argument to obtain more information about each
command and its arguments.

``python3 -m d3m index search``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Searches locally available primitives. Lists registered Python paths for
primitives installed on the system.

``python3 -m d3m index discover``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discovers primitives available on PyPi. Lists package names containing
D3M primitives on PyPi.

``python3 -m d3m index describe``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates a JSON description of a primitive.
