.. _metadata:

Metadata for primitives and the values they process
===================================================

Metadata is a core component of any data-based system. This repository
is standardizing how we represent metadata in the D3M program and
focusing on three types of metadata: \* metadata associated with
primitives \* metadata associated with datasets \* metadata associated
with values passed inside pipelines

This repository is also standardizing types of values being passed
between primitives in pipelines. While theoretically any value could be
passed between primitives, limiting them to a known set of values can
make primitives more compatible, efficient, and values easier to
introspect by TA3 systems.

.. _container_types:

Container types
---------------

All input and output (container) values passed between primitives should
expose a ``Sequence``
`protocol <https://www.python.org/dev/peps/pep-0544/>`__ (sequence in
samples) and provide ``metadata`` attribute with metadata.

``d3m.container`` module exposes such standard types:

-  ``Dataset`` – a class representing datasets, including D3M datasets,
   implemented in
   :mod:`d3m.container.dataset` module
-  ``DataFrame`` –
   :class:`pandas.DataFrame`
   with support for ``metadata`` attribute, implemented in
   :mod:`d3m.container.pandas` module
-  ``ndarray`` –
   :class:`numpy.ndarray`
   with support for ``metadata`` attribute, implemented in
   :mod:`d3m.container.numpy` module
-  ``List`` – a standard :class:`list` with support for ``metadata``
   attribute, implemented in
   :mod:`d3m.container.list` module

``List`` can be used to create a simple list container.

It is strongly encouraged to use the :class:`~d3m.container.pandas.DataFrame` container type for
primitives which do not have strong reasons to use something else
(:class:`~d3m.container.dataset.Dataset`\ s to operate on initial pipeline input, or optimized
high-dimensional packed data in :class:`~numpy.ndarray`\ s, or :class:`list`\ s to pass as
values to hyper-parameters). This makes it easier to operate just on
columns without type casting while the data is being transformed to make
it useful for models.

When deciding which container type to use for inputs and outputs of a
primitive, consider as well where an expected place for your primitive
is in the pipeline. Generally, pipelines tend to have primitives
operating on :class:`~d3m.container.dataset.Dataset` at the beginning, then use :class:`~d3m.container.pandas.DataFrame` and
then convert to :class:`~numpy.ndarray`.

.. _data_types:

Data types
----------

Container types can contain values of the following types:

* container types themselves
* Python builtin primitive types:

  * ``str``
  * ``bytes``
  * ``bool``
  * ``float``
  * ``int``
  * ``dict`` (consider using :class:`typing.Dict`, :class:`typing.NamedTuple`, or :ref:`TypedDict <mypy:typeddict>`)
  * ``NoneType``

Metadata
--------

:mod:`d3m.metadata.base` module provides a
standard Python implementation for metadata object.

When thinking about metadata, it is useful to keep in mind that metadata
can apply to different contexts:

* primitives
* values being passed
  between primitives, which we call containers (and are container types)
* datasets are a special case of a container
* to parts of data
  contained inside a container
* for example, a cell in a table can have
  its own metadata

Containers and their data can be seen as multi-dimensional structures.
Dimensions can have numeric (arrays) or string indexes (string to value
maps, i.e., dicts). Moreover, even numeric indexes can still have names
associated with each index value, e.g., column names in a table.

If a container type has a concept of *shape*
(:attr:`DataFrame.shape <pandas.DataFrame.shape>`, :attr:`ndarray.shape <numpy.ndarray.shape>`),
dimensions go in that order. For tabular data and existing container
types this means that the first dimension of a container is always
traversing samples (e.g., rows in a table), and the second dimension
columns.

Values can have nested other values and metadata dimensions go over all
of them until scalar values. So if a Pandas DataFrame contains
3-dimensional ndarrays, the whole value has 5 dimensions: two for rows
and columns of the DataFrame (even if there is only one column), and 3
for the array.

To tell to which part of data contained inside a container metadata
applies, we use a *selector*. Selector is a tuple of strings, integers,
or special values. Selector corresponds to a series of ``[...]`` item
getter Python operations on most values, except for Pandas DataFrame
where it corresponds to
:attr:`iloc <pandas.DataFrame.iloc>`
position-based selection.

Special selector values:

-  ``ALL_ELEMENTS`` – makes metadata apply to all elements in a given
   dimension (a wildcard)

Metadata itself is represented as a (potentially nested) dict. If
multiple metadata dicts comes from different selectors for the same
resolved selector location, they are merged together in the order from
least specific to more specific, later overriding earlier. ``null``
metadata value clears the key specified from a less specific selector.

Example
~~~~~~~

To better understand how metadata is attached to various parts of the
value, A `simple tabular D3M
dataset <https://gitlab.com/datadrivendiscovery/tests-data/tree/master/datasets/iris_dataset_1>`__
could be represented as a multi-dimensional structure:

.. code:: yaml

    {
      "0": [
        [0, 5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
        [1, 4.9, 3, 1.4, 0.2, "Iris-setosa"],
        ...
      ]
    }

It contains one resource with ID ``"0"`` which is the first dimension
(using strings as index; it is a map not an array), then rows, which is
the second dimension, and then columns, which is the third dimension.
The last two dimensions are numeric.

In Python, accessing third column of a second row would be
``["0"][1][2]`` which would be value ``3``. This is also the selector if
we would want to attach metadata to that cell. If this metadata is
description for this cell, we can thus describe this datum metadata as a
pair of a selector and a metadata dict:

-  selector: ``["0"][1][2]``
-  metadata:
   ``{"description": "Measured personally by Ronald Fisher."}``

Dataset-level metadata have empty selector:

-  selector: ``[]``
-  metadata: ``{"id": "iris_dataset_1", "name": "Iris Dataset"}``

To describe first dimension itself, we set ``dimension`` metadata on the
dataset-level (container). ``dimension`` describes the next dimension at
that location in the data structure.

-  selector: ``[]``
-  metadata: ``{"dimension": {"name": "resources", "length": 1}}``

This means that the full dataset-level metadata is now:

.. code:: json

    {
      "id": "iris_dataset_1",
      "name": "Iris Dataset",
      "dimension": {
        "name": "resources",
        "length": 1
      }
    }

To attach metadata to the first (and only) resource, we can do:

-  selector: ``["0"]``
-  metadata:
   ``{"structural_type": "pandas.core.frame.DataFrame", "dimension": {"length": 150, "name": "rows"}``

``dimension`` describes rows.

Columns dimension:

-  selector: ``["0"][ALL_ELEMENTS]``
-  metadata: ``{"dimension": {"length": 6, "name": "columns"}}``

Observe that there is no requirement that dimensions are aligned from
the perspective of metadata. But in this case they are, so we can use
``ALL_ELEMENTS`` wildcard to describe columns for all rows.

Third column metadata:

-  selector: ``["0"][ALL_ELEMENTS][2]``
-  metadata:
   ``{"name": "sepalWidth", "structural_type": "builtins.str", "semantic_types": ["http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute"]}``

Column names belong to each particular column and not all columns. Using
``name`` can serve to assign a string name to otherwise numeric
dimension.

We attach names and types to datums themselves and not dimensions.
Because we use ``ALL_ELEMENTS`` selector, this is internally stored
efficiently. We see traditional approach of storing this information in
the header of a column as a special case of a ``ALL_ELEMENTS`` selector.

Note that the name of a column belongs to the metadata because it is
just an alternative way to reference values in an otherwise numeric
dimension. This is different from a case where a dimension has
string-based index (a map/dict) where names of values are part of the
data structure at that dimension. Which approach is used depends on the
structure of the container for which metadata is attached to.

Default D3M dataset loader found in this package parses all tabular
values as strings and add semantic types, if known, for what could those
strings be representing (a float) and its role (an attribute). This
allows primitives later in a pipeline to convert them to proper
structural types but also allows additional analysis on original values
before such conversion is done.

Fetching all metadata for ``["0"][1][2]`` now returns:

.. code:: json

    {
      "name": "sepalWidth",
      "structural_type": "builtins.str",
      "semantic_types": [
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/Attribute"
      ],
      "description": "Measured personally by Ronald Fisher."
    }

.. _metadata_api:

API
~~~

:mod:`d3m.metadata.base` module provides two
classes which serve for storing metadata on values: :class:`~d3m.metadata.base.DataMetadata` for
data values, and :class:`~d3m.metadata.base.PrimitiveMetadata` for primitives. It also exposes a
:const:`~d3m.metadata.base.ALL_ELEMENTS` constant to be used in selectors.

You can see public methods available on classes documented in their
code. Some main ones are:

-  ``__init__(metadata)`` – constructs a new instance of the metadata
   class and optionally initializes it with top-level metadata
-  ``update(selector, metadata)`` – updates metadata at a given location
   in data structure identified by a selector
-  ``query(selector)`` – retrieves metadata at a given location
-  ``query_with_exceptions(selector)`` – retrieves metadata at a given
   location, but also returns metadata for selectors which have metadata
   which differs from that of ``ALL_ELEMENTS``
-  ``remove(selector)`` – removes metadata at a given location
-  ``get_elements(selector)`` – lists element names which exists at a
   given location
-  ``to_json()`` – converts metadata to a JSON representation
-  ``pretty_print()`` – pretty-print all metadata

``PrimitiveMetadata`` differs from ``DataMetadata`` that it does not
accept selector in its methods because there is no structure in
primitives.

Standard metadata keys
~~~~~~~~~~~~~~~~~~~~~~

You can use custom keys for metadata, but the following keys are
standardized, so you should use those if you are trying to represent the
same metadata:
https://metadata.datadrivendiscovery.org/schemas/v0/definitions.json

The same key always have the same meaning and we reuse the same key in
different contexts when we need the same meaning. So instead of having
both ``primitive_name`` and ``dataset_name`` we have just ``name``.

Different keys are expected in different contexts:

-  ``primitive`` –
   https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json
-  ``container`` –
   https://metadata.datadrivendiscovery.org/schemas/v0/container.json
-  ``data`` –
   https://metadata.datadrivendiscovery.org/schemas/v0/data.json

A more user friendly visualization of schemas listed above is available
at https://metadata.datadrivendiscovery.org/.

Contribute: Standardizing metadata schemas are an ongoing process. Feel
free to contribute suggestions and merge requests with improvements.

.. _primitive-metadata:

Primitive metadata
~~~~~~~~~~~~~~~~~~

Part of primitive metadata can be automatically obtained from
primitive's code, some can be computed through evaluation of primitives,
but some has to be provided by primitive's author. Details of which
metadata is currently standardized and what values are possible can be
found in primitive's JSON schema. This section describes author's
metadata into more detail. Example of primitive's metadata provided by
an author from `Monomial test
primitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py#L32>`__,
slightly modified:

.. code:: python

    metadata = metadata_module.PrimitiveMetadata({
        'id': '4a0336ae-63b9-4a42-860e-86c5b64afbdd',
        'version': '0.1.0',
        'name': "Monomial Regressor",
        'keywords': ['test primitive'],
        'source': {
            'name': 'Test team',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py',
                'https://gitlab.com/datadrivendiscovery/tests-data.git',
            ],
        },
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@{git_commit}#egg=test_primitives&subdirectory=primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'location_uris': [
            'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        ],
        'python_path': 'd3m.primitives.test.MonomialPrimitive',
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.LINEAR_REGRESSION,
        ],
        'primitive_family': metadata_module.PrimitiveFamily.REGRESSION,
    })

-  Primitive's metadata provided by an author is defined as a class
   attribute and instance of :class:`~d3m.metadata.base.PrimitiveMetadata`.
-  When class is defined, class is automatically analyzed and metadata
   is extended with automatically obtained values from class code.
-  ``id`` can be simply generated using :func:`uuid.uuid4` in Python and
   should never change. **Do not reuse IDs and do not use the ID from
   this example.**
-  When primitive's code changes you should update the version, a `PEP
   440 <https://www.python.org/dev/peps/pep-0440/>`__ compatible one.
   Consider updating a version every time you change code, potentially
   using `semantic versioning <https://semver.org/>`__, but nothing of
   this is enforced.
-  ``name`` is a human-friendly name of the primitive.
-  ``keywords`` can be anything you want to convey to users of the
   primitive and which could help with primitive's discovery.
-  ``source`` describes where the primitive is coming from. The required
   value is ``name`` to tell information about the author, but you might
   be interested also in ``contact`` where you can put an e-mail like
   ``mailto:author@example.com`` as a way to contact the author.
   ``uris`` can be anything. In above, one points to the code in GitLab,
   and another to the repo. If there is a website for the primitive, you
   might want to add it here as well. These URIs are not really meant
   for automatic consumption but are more as a reference. See
   ``location_uris`` for URIs to the code.
-  ``installation`` is important because it describes how can your
   primitive be automatically installed. Entries are installed in order
   and currently the following types of entries are supported:
-  A ``PIP`` package available on PyPI or some other package registry:

   ::

       ```
       {
         'type': metadata_module.PrimitiveInstallationType.PIP,
         'package': 'my-primitive-package',
         'version': '0.1.0',
       }
       ```

-  A ``PIP`` package available at some URI. If this is a git repository,
   then an exact git hash and ``egg`` name should be provided. ``egg``
   name should match the package name installed. Because here we have a
   chicken and an egg problem: how can one commit a hash of code version
   if this changes the hash, you can use a helper utility function to
   provide you with a hash automatically at runtime. ``subdirectory``
   part of the URI suffix is not necessary and is here just because this
   particular primitive happens to reside in a subdirectory of the
   repository.
-  A ``DOCKER`` image which should run while the primitive is operating.
   Starting and stopping of a Docker container is managed by a caller,
   which passes information about running container through primitive's
   ``docker_containers`` ``__init__`` argument. The argument is a
   mapping between the ``key`` value and address and ports at which the
   running container is available. See `Sum test
   primitive <https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py#L66>`__
   for an example:

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.DOCKER,
           'key': 'summing',
           'image_name': 'registry.gitlab.com/datadrivendiscovery/tests-data/summing',
           'image_digest': 'sha256:07db5fef262c1172de5c1db5334944b2f58a679e4bb9ea6232234d71239deb64',
       }
       ```

-  A ``UBUNTU`` entry can be used to describe a system library or
   package required for installation or operation of your primitive. If
   your other dependencies require a system library to be installed
   before they can be installed, list this entry before them in
   ``installation`` list.

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.UBUNTU,
           'package': 'ffmpeg',
           'version': '7:3.3.4-2',
       }
       ```

-  A ``FILE`` entry allows a primitive to specify a static file
   dependency which should be provided by a caller to a primitive.
   Caller passes information about the file path of downloaded file
   through primitive's ``volumes`` ``__init__`` argument. The argument
   is a mapping between the ``key`` value and file path. The filename
   portion of the provided path does not necessary match the filename
   portion of the file's URI.

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.FILE,
           'key': 'model',
           'file_uri': 'http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/googlenet_finetune_web_car_iter_10000.caffemodel',
           'file_digest': '6bdf72f703a504cd02d7c3efc6c67cbbaf506e1cbd9530937db6a698b330242e',
       }
       ```

-  A ``TGZ`` entry allows a primitive to specify a static directory
   dependency which should be provided by a caller to a primitive.
   Caller passes information about the directory path of downloaded and
   extracted file through primitive's ``volumes`` ``__init__`` argument.
   The argument is a mapping between the ``key`` value and directory
   path.

   ::

       ```
       {
           'type': metadata_module.PrimitiveInstallationType.TGZ,
           'key': 'mails',
           'file_uri': 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz',
           'file_digest': 'b3da1b3fe0369ec3140bb4fbce94702c33b7da810ec15d718b3fadf5cd748ca7',
       }
       ```

-  If you can provide, ``location_uris`` points to an exact code used by
   the primitive. This can be obtained through installing a primitive,
   but it can be helpful to have an online resource as well.
-  ``python_path`` is a path under which the primitive will get mapped
   through ``setup.py`` entry points. This is very important to keep in
   sync.
-  ``algorithm_types`` and ``primitive_family`` help with discovery of a
   primitive. They are required and if suitable values are not available
   for you, make a merge request and propose new values. As you see in
   the code here and in ``installation`` entries, you can use directly
   Python enumerations to populate these values.

Some other metadata you might be interested to provide to help callers
use your primitive better are ``preconditions`` (what preconditions
should exist on data for primitive to operate well), ``effects`` (what
changes does a primitive do to data), and a ``hyperparams_to_tune`` hint
to help callers know which hyper-parameters are most important to focus
on.

Primitive metadata also includes descriptions of a primitive and its
methods. These descriptions are automatically obtained from primitive's
docstrings. Docstrings should be made according to :ref:`numpy docstring
format <numpy:format>`
(`examples <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__).

Data metadata
~~~~~~~~~~~~~

Every value passed around a pipeline has metadata associated with it.
Defined container types have an attribute ``metadata`` to contain it.
API available to manipulate metadata is still evolving because many
operations one can do on data are reasonable also on metadata (e.g.,
slicing and combining data). Currently, every operation on data clears
and re-initializes associated metadata.

    **Note:** While part of primitive's metadata is obtained
    automatically nothing like that is currently done for data metadata.
    This means one has to manually populate with dimension and typing
    information. This will be improved in the future with automatic
    extraction of this metadata from data.

Parameters
----------

A base class to be subclassed and used as a type for :class:`~d3m.metadata.params.Params` type
argument in primitive interfaces can be found in the
:mod:`d3m.metadata.params` module. An
instance of this subclass should be returned from primitive's
:meth:`~d3m.metadata.params.Params.get_params` method, and accepted in :meth:`~d3m.metadata.params.Params.set_params`.

To define parameters a primitive has you should subclass this base class
and define parameters as class attributes with type annotations.
Example:

.. code:: python

    import numpy
    from d3m.metadata import params

    class Params(params.Params):
        weights: numpy.ndarray
        bias: float

:class:`~d3m.metadata.params.Params` class is just a fancy Python dict which checks types of
parameters and requires all of them to be set. You can create it like:

.. code:: python

    ps = Params({'weights': weights, 'bias': 0.1})
    ps['bias']

::

    0.01

``weights`` and ``bias`` do not exist as an attributes on the class or
instance. In the class definition, they are just type annotations to
configure which parameters are there.

    **Note:** :class:`~d3m.metadata.params.Params` class uses ``parameter_name: type`` syntax
    while :class:`~d3m.metadata.hyperparams.Hyperparams` class uses
    ``hyperparameter_name = Descriptor(...)`` syntax. Do not confuse
    them.

.. _hyperparameters:

Hyper-parameters
----------------

A base class for hyper-parameters description for primitives can be
found in the
:mod:`d3m.metadata.hyperparams` module.

To define a hyper-parameters space you should subclass this base class
and define hyper-parameters as class attributes. Example:

.. code:: python

    from d3m.metadata import hyperparams

    class Hyperparams(hyperparams.Hyperparams):
        learning_rate = hyperparams.Uniform(lower=0.0, upper=1.0, default=0.001, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ])
        clusters = hyperparams.UniformInt(lower=1, upper=100, default=10, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ])

To access hyper-parameters space configuration, you can now call:

.. code:: python

    Hyperparams.configuration

::

    OrderedDict([('learning_rate', Uniform(lower=0.0, upper=1.0, q=None, default=0.001)), ('clusters', UniformInt(lower=1, upper=100, default=10))])

To get a random sample of all hyper-parameters, call:

.. code:: python

    hp1 = Hyperparams.sample(random_state=42)

::

    Hyperparams({'learning_rate': 0.3745401188473625, 'clusters': 93})

To get an instance with all default values:

.. code:: python

    hp2 = Hyperparams.defaults()

::

    Hyperparams({'learning_rate': 0.001, 'clusters': 10})

:class:`~d3m.metadata.hyperparams.Hyperparams` class is just a fancy read-only Python dict. You can
also manually create its instance:

.. code:: python

    hp3 = Hyperparams({'learning_rate': 0.01, 'clusters': 20})
    hp3['learning_rate']

::

    0.01

If you want to use most of default values, but set some, you can thus
use this dict-construction approach:

.. code:: python

    hp4 = Hyperparams(Hyperparams.defaults(), clusters=30)

::

    Hyperparams({'learning_rate': 0.001, 'clusters': 30})

There is no class- or instance-level attribute ``learning_rate`` or
``clusters``. In the class definition, they were used only for defining
the hyper-parameters space, but those attributes were extracted out and
put into ``configuration`` attribute.

There are four types of hyper-parameters: \* tuning parameters which
should be tuned during hyper-parameter optimization phase \* control
parameters which should be determined during pipeline construction phase
and are part of the logic of the pipeline \* parameters which control
the use of resources by the primitive \* parameters which control which
meta-features are computed by the primitive

You can use hyper-parameter's semantic type to differentiate between
those types of hyper-parameters using the following URIs:

* https://metadata.datadrivendiscovery.org/types/TuningParameter
* https://metadata.datadrivendiscovery.org/types/ControlParameter
* https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter
* https://metadata.datadrivendiscovery.org/types/MetafeatureParameter

Once you define a :class:`~d3m.metadata.hyperparams.Hyperparams` class for your primitive you can pass
it as a class type argument in your primitive's class definition:

.. code:: python

    class MyPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
        ...

Those class type arguments are then automatically extracted from the
class definition and made part of primitive's metadata. This allows the
caller to access the :class:`~d3m.metadata.hyperparams.Hyperparams` class to crete an instance to pass
to primitive's constructor:

.. code:: python

    hyperparams_class = MyPrimitive.metadata.get_hyperparams()
    primitive = MyPrimitive(hyperparams=hyperparams_class.defaults())

    **Note:** :class:`~d3m.metadata.hyperparams.Hyperparams` class uses
    ``hyperparameter_name = Descriptor(...)`` syntax while :class:`~d3m.metadata.params.Params`
    class uses ``parameter_name: type`` syntax. Do not confuse them.

Problem description
-------------------

:mod:`d3m.metadata.problem` module provides
a parser for problem description into a normalized Python object.

You can load a problem description and get the loaded object dumped back
by running:

.. code:: bash

    python3 -m d3m problem describe <path to problemDoc.json>

Dataset
-------

This package also provides a Python class to load and represent datasets
in Python in :mod:`d3m.container.dataset`
module. This container value can serve as an input to the whole pipeline
and be used as input for primitives which operate on a dataset as a
whole. It allows one to register multiple loaders to support different
formats of datasets. You pass an URI to a dataset and it automatically
picks the right loader. By default it supports:

-  D3M dataset. Only ``file://`` URI scheme is supported and URI should
   point to the ``datasetDoc.json`` file. Example:
   ``file:///path/to/datasetDoc.json``
-  CSV file. Many URI schemes are supported, including remote ones like
   ``http://``. URI should point to a file with ``.csv`` extension.
   Example: ``http://example.com/iris.csv``
-  Sample datasets from :mod:`sklearn.datasets`.
   Example: ``sklearn://boston``

You can load a dataset and get the loaded object dumped back by running:

.. code:: bash

    python3 -m d3m dataset describe <path to the dataset file>
