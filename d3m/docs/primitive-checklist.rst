.. _primitive-good-citizen:

Primitive Good Citizen Checklist
================================

This is a list of dos, don'ts and things to consider when crafting a new primitive or updating an existing one. This
list is not exhaustive so please add new items to the list as they are discovered! An example of a primitive that
endeavors to adheres to all of the following guidance can be found `here`_:

DO's

* Do complete the documentation on the primitive such as:

  * Primitive family, algorithm type.
  * Docstring of the primitive's Python class.

    * One line summary first:

      * Primitive name should be close to this.
      * Primitive path should be close to this as well.

    * Longer documentation/description after, all in the main docstring of the class.

  * Provide pipeline examples together with the primitive annotation.
  * Docstrings in `numpy style`_.
  * Please use `reStructuredText`_ instead of markdown or other formats.
  * Maintain a change-log of alterations to the primitive (somewhere in the primitive's repo, consider using a `standard format`_).
  * One should also add point of contact information and the git repository link in primitive's metadata
    (``source.name``, ``source.contact`` and ``source.uris`` metadata fields).
  * Add your primitive name to the `list of primitive names`_ if it does not already
    exist. Chances are that your generic primitive name is in that list and you should use that name for your primitive.

* Do annotate your Primitive with Python types.

* Do make sure the output from your produce method is a d3m container type.



* If your primitive is operating on columns and rows:

  * Do include ``d3mIndex`` column in produced output if input has ``d3mIndex`` column.
    * You can make this behavior controlled by the ``add_index_columns`` hyper-parameter.

  * If a primitive has a hyper-paramer to directly set which columns to operate on, do use column
    indices and not column names to identify those columns.

    * Consider using a pair of hyper-parameters: ``use_columns`` and ``exclude_columns`` with standard logic.

  * When deciding on which columns to operate, when using semantic types, do use
    ``https://metadata.datadrivendiscovery.org/types/TrueTarget``
    and ``https://metadata.datadrivendiscovery.org/types/RedactedPrivilegedData`` semantic types and not
    ``https://metadata.datadrivendiscovery.org/types/SuggestedTarget`` and
    ``https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData``.
    The latter are semantic types which come from the dataset, the former are those which come from the problem description.
    While it is true that currently generally they always match, in fact primitives should just respect those coming from
    the problem description. The dataset has them so that one can create problem descriptions on the fly, if needed.

* Be mindful that data being passed through a pipeline also has metadata:

  * If your primitive generates new data (e.g., new columns), add metadata suitable for those columns:

    * Name the column appropriately for human consumption by setting column's ``name`` metadata.

    * Set semantic types appropriately.

    * If your primitive is producing target predictions, add ``https://metadata.datadrivendiscovery.org/types/PredictedTarget``
      to a column containing those predictions.

    * Remember metadata encountered on target columns during fitting, and reuse that metadata as much
      as reasonable when producing target predictions.

  * If your primitive is transforming existing data (e.g., transforming columns), reuse as much metadata from
    original data as reasonable, but do update metadata based on new data.

    * If structural type of the column changes, make sure you note this change in metadata as well.

  * Support also non-standard metadata and try to pass it through as-is if possible.

* Do write unit tests for your primitives. This greatly aids porting to a new version of the core package.

  * Test pickle and unpickle of the primitive (both fitted and unfitted primitives).
  * Test with use of semantic types to select columns to operate on, and without the use of semantic types.
  * Test with all return types: ``append``, ``replace``, ``new``.
  * Test all hyper-parameter values with their ``sample`` method.
  * Use/contribute to `tests data repository`_.

* Do clearly define hyper-parameters (bounds, descriptions, semantic types).

  * Suggest new classes of hyper-parameters if needed.
  * Consider if ``upper_inclusive`` and ``lower_inclusive`` values should be included or not for every hyper-parameter
  * Define reasonable hyper-parameters which can be automatically populated/searched by TA2.
    A hyper-parameter such as ``hyperparams.Hyperparameter[typing.Sequence[Any]]`` is not useful in this case.
  * Ensure that your primitive can be run successfully with default settings for all hyper-parameters.
  * If there are combinations of hyper-parameters settings that are suboptimal please note this in the documentation. For
    example: "If hyper-parameter A is set to a True, hyper-parameter B must always be a positive integer".

* Do bump primitive version when changing hyper-parameters, method signatures or params.
  In short, on any API change of your primitive.

* If your primitive can use GPUs if available, set ``can_use_gpus`` primitive's metadata to true.

* If your primitive can use different number of CPUs/cores, expose a hyper-parameter with semantic types
  `https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter` and `https://metadata.datadrivendiscovery.org/types/CPUResourcesUseParameter`
  and allow caller to control the number of CPUs/cores used through it.

  * Make sure that the default value of such hyper-parameter is 1.

DON'Ts

* Don't change the input DataFrame! Make a copy and make changes to the copy instead. The original input DataFrame is
  assumed never to change between primitives in the pipeline.
* Don't return DataFrames with a (non-default) Pandas DataFrame index. It can be utilized internally, but drop it before
  returning. On output a default index should be provided.

PLEASE CONSIDER

* Consider using/supporting semantic types to select which columns to operate on, and use the `use_semantic_types` hyper-parameter.
* Consider allowing three types of outputs strategies: ``new``/``append``/``replace`` output, if operating on columns,
  controlled by the ``return_result`` hyper-parameter.
* Consider picking the input and output format/structure of data to match other primitives of the same family/type. If
  necessary, convert data to the format you need inside your primitive. Pipelines tend to start with datasets, then go
  to dataframes, and then to ndarrays sometimes, returning predictions as a dataframe.
  Consider where your primitive in a pipeline generally should be and
  consider that when deciding on what are inputs and outputs of your primitive. Consider that your primitive will be
  chosen dynamically by a TA2 and will be expected to behave in predictable ways based on family and base class.
* Consider using a specific hyper-parameter class instead of the hyper-parameter base class as it is not very useful for
  TA2s. For example use ``hyperparams.Set`` instead of ``hyperparams.Hyperparameter[typing.Sequence[Any]]``. It is
  better to use the former as it is far more descriptive.
* Use a base class for your primitive which makes sense based on semantics of the base class and not necessarily
  how a human would understand the primitive.
* Consider that your primitive will be chosen dynamically by a TA2 and will
  be expected to behave in predictable ways based on primitive family and base class.

.. _here: https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/random_forest.py
.. _numpy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _tests data repository: https://gitlab.com/datadrivendiscovery/tests-data
.. _standard format: https://keepachangelog.com/en/1.0.0/
.. _list of primitive names: https://gitlab.com/datadrivendiscovery/d3m/-/blob/devel/d3m/metadata/primitive_names.py
