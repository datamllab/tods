## v2020.5.18

### Enhancements

* Scoring primitive and pipeline now accept new hyper-parameter `all_labels`
  which can be used to provide information about all labels possible in a target
  column.
  [#431](https://gitlab.com/datadrivendiscovery/d3m/-/issues/431)
  [!377](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/377)
* Added `all_distinct_values` metadata field which can contain all values (labels)
  which can be in a column. This is meant to be used on target columns to help
  implementing `ContinueFitMixin` in a primitive which might require knowledge
  of all possible labels before starting fitting on a subset of data.
  [#447](https://gitlab.com/datadrivendiscovery/d3m/-/issues/447)
  [!377](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/377)
* Reference runtime now does not keep primitive instances in memory anymore
  but uses `get_params`/`set_params` to retain and reuse only primitive's parameters.
  This makes memory usage lower and allows additional resource releasing when primitive's
  object is freed (e.g., releasing GPUs).
  [#313](https://gitlab.com/datadrivendiscovery/d3m/-/issues/313)
  [!376](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/376)
* Added support for version 4.1.0 of D3M dataset schema:
  * Added `MONTHS` to column's `time_granularity` metadata.
    [!340](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/340)
  * Added mean reciprocal rank and hits at k metrics.
    [!361](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/361)
  * Added `https://metadata.datadrivendiscovery.org/types/Rank` semantic type
    and `rank_for` metadata field. `PerformanceMetric` classes have now
    `requires_rank` method.
    [!372](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/372)
  * Added `NESTED` task keyword.
    [!372](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/372)
  * Added `file_columns_count` metadata field and updated `file_columns` metadata field
    with additional sub-fields. Also renamed sub-field `name` to `column_name` and added
    `column_index` sub-fields to `file_columns` metadata.
    [!372](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/372)    
    **Backwards incompatible.**
* Moved high-level primitive base classes for file readers and dataset splitting
  from common primitives to d3m core package.
  [!120](https://gitlab.com/datadrivendiscovery/common-primitives/-/merge_requests/120)
  [!339](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/339)
* A warning is issued if a primitive uses a global random source
  during pipeline execution. Such behavior can make pipeline
  execution not reproducible.
  [#384](https://gitlab.com/datadrivendiscovery/d3m/-/issues/384)
  [!365](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/365)
* CLI accepts `--logging-level` argument to configure which logging
  messages are printed to the console.
  [!360](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/360)
* Output to stdout/stderr during pipeline execution is now not suppressed
  anymore, which makes it possible to debug pipeline execution using pdb.
  Stdout/stderr is at the same time still logged to Python logging.
  [#270](https://gitlab.com/datadrivendiscovery/d3m/-/issues/270)
  [!360](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/360)
* Redirect from stdout to Python logging now operates per lines and
  not per write operations, makes logs more readable.
  [#168](https://gitlab.com/datadrivendiscovery/d3m/-/issues/168)
  [!358](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/358)
* Made sure that multi-label metrics work correctly.
  [#370](https://gitlab.com/datadrivendiscovery/d3m/-/issues/370)
  [!343](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/343)
* Implemented ROC AUC metrics. They require predictions to include
  confidence for all possible labels.
  [#317](https://gitlab.com/datadrivendiscovery/d3m/-/issues/317)
  [!318](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/318)
* Additional (non-standard) performance metrics can now be registered
  using `PerformanceMetric.register_metric` class method.
  [#207](https://gitlab.com/datadrivendiscovery/d3m/-/issues/207)
  [!348](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/348)
* All D3M enumerations can now be extended with additional values
  through `register_value` class method. This allows one to add values
  to existing standard values (which come from the metadata schema).
  Internally, enumeration values are now represented as strings and not
  integers anymore.
  [#438](https://gitlab.com/datadrivendiscovery/d3m/-/issues/438)
  [!348](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/348)
  **Could be backwards incompatible.**
* Added CLI to validate primitive descriptions for metalearning database
  (`python3 -m d3m index validate`).
  [!333](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/333)
* Raise an exception during dataset loading if `targets.csv` file does
  not combine well with the dataset entry point.
  [!330](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/330)

### Bugfixes

* CLI now displays correct error messages for invalid arguments to subcommands.
  [#409](https://gitlab.com/datadrivendiscovery/d3m/-/issues/409)
  [!368](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/368)
* Reference runtime does not call `fit` and `produce`
  methods in a loop anymore. This mitigates an infinite loop for misbehaving primitives.
  [!364](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/364)
* During pipeline execution all Python logging is now recorded in the
  pipeline run and it does not depend anymore on logging level otherwise
  configured during execution.
  [!360](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/360)
* Default sampling code for hyper-parameters now makes sure to return
  values in original types and not numpy ones.
  [#440](https://gitlab.com/datadrivendiscovery/d3m/-/issues/440)
  [!352](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/352)
* We now ensure that file handles opened for CLI commands are flushed
  so that data is not lost.
  [#436](https://gitlab.com/datadrivendiscovery/d3m/issues/436)
  [!335](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/335)
* Fixed saving exposed produced outputs for `fit-score` CLI command when
  scoring failed.
  [!341](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/341)
* Made sure `time_granularity` metadata is saved when saving a D3M dataset.
  [!340](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/340)
* Changed version of GitPython dependency to 3.1.0 to fix older versions
  being broken because of its own unconstrained upper dependency.
  [!336](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/336)
* Fixed how paths are constructed when exposing and saving produced values.
  [!336](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/336)

### Other

* Added guides to the documentation.
  [!351](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/351)
  [!374](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/374)
* Removed type annotations from docstrings. Python type annotations are now used instead when rendering documentation.
  [#239](https://gitlab.com/datadrivendiscovery/d3m/-/issues/239)
  [!371](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/371)
* Renamed `blacklist` in `d3m.index.load_all` and `primitives_blacklist` in `d3m.metadata.pipeline.Resolver`
  to `blocklist` and `primitives_blocklist`, respectively.
  **Backwards incompatible.**
* Removed `https://metadata.datadrivendiscovery.org/types/GPUResourcesUseParameter`
  semantic type. Added `can_use_gpus` primitive metadata field to signal that
  the primitive can use GPUs if available.
  [#448](https://gitlab.com/datadrivendiscovery/d3m/-/issues/448)
  [!369](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/369)
  **Backwards incompatible.**
* Clarified that hyper-parameters using `https://metadata.datadrivendiscovery.org/types/CPUResourcesUseParameter`
  should have 1 as default value.
  [!369](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/369)
* Clarified that it is not necessary to call `fit` before calling
  `continue_fit`.
* `index` CLI command has been renamed to `primitive` CLI command.
  [#437](https://gitlab.com/datadrivendiscovery/d3m/-/issues/437)
  [!363](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/363)
* `numpy.matrix` has been removed as an allowed container type, as it
  was deprecated by NumPy.
  [#230](https://gitlab.com/datadrivendiscovery/d3m/-/issues/230)
  [!362](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/362)
  **Backwards incompatible.**  
* CLI has now `--version` command which returns the version of the d3m
  core package itself.
  [#378](https://gitlab.com/datadrivendiscovery/d3m/-/issues/378)
  [!359](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/359)
* Upgraded schemas to JSON Schema draft 7, and upgraded Python `jsonschema`
  dependency to version 3.
  [#392](https://gitlab.com/datadrivendiscovery/d3m/-/issues/392)
  [!342](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/342)
* Added a Primitive Good Citizen Checklist to documentation, documenting
  some best practices when writing a primitive.
  [#127](https://gitlab.com/datadrivendiscovery/d3m/-/issues/127)
  [!347](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/347)
  [!355](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/355)
* Updated upper bounds of core dependencies to latest available versions.
  [!337](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/337)
* Added to `algorithm_types`:
    * `SAMPLE_SELECTION`
    * `SAMPLE_MERGING`
    * `MOMENTUM_CONTRAST`
    * `CAUSAL_ANALYSIS`

    [!332](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/332)
    [!357](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/357)
    [!373](https://gitlab.com/datadrivendiscovery/d3m/-/merge_requests/373)

## v2020.1.9

### Enhancements

* Support for D3M datasets with minimal metadata.
  [#429](https://gitlab.com/datadrivendiscovery/d3m/issues/429)
  [!327](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/327)
* Pipeline runs (and in fact many other input documents) can now be directly used gzipped
  in all CLI commands. They have to have filename end with `.gz` for decompression to happen
  automatically.
  [#420](https://gitlab.com/datadrivendiscovery/d3m/issues/420)
  [!317](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/317)
* Made problem descriptions again more readable when converted to JSON.
  [!316](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/316)
* Improved YAML handling to encourage faster C implementation.
  [#416](https://gitlab.com/datadrivendiscovery/d3m/issues/416)
  [!313](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/313)

### Bugfixes

* Fixed the error message if all required CLI arguments are not passed to the runtime.
  [#411](https://gitlab.com/datadrivendiscovery/d3m/issues/411)
  [!319](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/319)
* Removed assumption that all successful pipeline run steps have method calls.
  [#422](https://gitlab.com/datadrivendiscovery/d3m/issues/422)
  [!320](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/320)
* Fixed "Duplicate problem ID" warnings when multiple problem descriptions
  have the same problem ID, but in fact they are the same problem description.
  No warning is made in this case anymore.
  [#417](https://gitlab.com/datadrivendiscovery/d3m/issues/417)
  [!321](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/321)
* Fixed the use of D3M container types in recent versions of Keras and TensorFlow.
  [#426](https://gitlab.com/datadrivendiscovery/d3m/issues/426)
  [!322](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/322)
* Fixed `validate` CLI commands to work on YAML files.

### Other

* Updated upper bounds of core dependencies to latest available versions.
  [#427](https://gitlab.com/datadrivendiscovery/d3m/issues/427)
  [!325](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/325)
* Refactored default pipeline run parser implementation to make it
  easier to provide alternative dataset and problem resolvers.
  [!314](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/314)
* Moved out local test primitives into [`tests/data` git submodule](https://gitlab.com/datadrivendiscovery/tests-data).
  Now all test primitives are in one place.
  [#254](https://gitlab.com/datadrivendiscovery/d3m/issues/254)
  [!312](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/312)

## v2019.11.10

* Support for version 4.0.0 of D3M dataset schema has been added.
* D3M core package now supports loading directly datasets from OpenML.
* When saving `Dataset` object to D3M dataset format, metadata is now preserved.
* NetworkX objects are not anymore container types and are not allowed
  anymore to be passed as values between primitives.
* "Meta" files are not supported anymore by the runtime. Instead save a
  pipeline run with configuration of the run you want, and use the pipeline
  run to re-run using that configuration.

### Enhancements

* Primitive family `REMOTE_SENSING` has been added.
  [!310](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/310)
* Added support for version 4.0.0 of D3M dataset schema:
    * There are no more `NODE` and `EDGE` references (used in graph datasets),
      but only `NODE_ATTRIBUTE` and `EDGE_ATTRIBUTE`.
    * `time_granularity` can now be present on a column.
    * `forecasting_horizon` can now be present in a problem description.
    * `task_type` and `task_subtype` have been merged into `task_keywords`.
      As a consequence, Python `TaskType` and `TaskSubtype` were replaced
      with `TaskKeyword`.

    [#401](https://gitlab.com/datadrivendiscovery/d3m/issues/401)
    [!310](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/310)
    **Backwards incompatible.**  

* Added OpenML dataset loader. Now you can pass an URL to a OpenML dataset
  and it will be downloaded and converted to a `Dataset` compatible object,
  with including many of available meta-features. Combined with support
  for saving datasets, this now allows easy conversion between OpenML
  datasets and D3M datasets, e.g., `python3 -m d3m dataset convert -i https://www.openml.org/d/61 -o out/datasetDoc.json`.
  [#252](https://gitlab.com/datadrivendiscovery/d3m/issues/252)
  [!309](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/309)
* When saving and loading D3M datasets, metadata is now preserved.
  [#227](https://gitlab.com/datadrivendiscovery/d3m/issues/227)
  [!265](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/265)
* Metadata can now be converted to a JSON compatible structure in a
  reversible manner.
  [#373](https://gitlab.com/datadrivendiscovery/d3m/issues/373)
  [!308](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/308)
* Pipeline run now records if a pipeline was run as a standard pipeline
  under `run.is_standard_pipeline` field.
  [#396](https://gitlab.com/datadrivendiscovery/d3m/issues/396)
  [!249](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/249)
* "meta" files have been replaced with support for rerunning pipeline runs.
  Instead of configuring a "meta" file with configuration how to run a
  pipeline, simply provide an example pipeline run which demonstrates how
  the pipeline was run. Runtime does not have `--meta` argument anymore,
  but has now `--input-run` argument instead.
  [#202](https://gitlab.com/datadrivendiscovery/d3m/issues/202)
  [!249](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/249)
  **Backwards incompatible.**  
* Changed `LossFunctionMixin` to support multiple loss functions.
  [#386](https://gitlab.com/datadrivendiscovery/d3m/issues/386)
  [!305](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/305)
  **Backwards incompatible.**  
* Pipeline equality and hashing functions now have `only_control_hyperparams`
  argument which can be set to use only control hyper-parameters when doing
  comparisons.
  [!289](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/289)
* Pipelines and other YAML files are now recognized with both `.yml` and
  `.yaml` file extensions.
  [#375](https://gitlab.com/datadrivendiscovery/d3m/issues/375)
  [!302](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/302)
* `F1Metric`, `F1MicroMetric`, and `F1MacroMetric` can now operate on
  multiple target columns and average scores for all of them.
  [#400](https://gitlab.com/datadrivendiscovery/d3m/issues/400)
  [!298](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/298)
* Pipelines and pipeline runs can now be serialized with Arrow.
  [#381](https://gitlab.com/datadrivendiscovery/d3m/issues/381)
  [!290](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/290)
* `describe` CLI commands now accept `--output` argument to control where
  their output is saved to.
  [!279](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/279)

### Bugfixes

* Made exposed outputs be stored even in the case of an exception.
  [#380](https://gitlab.com/datadrivendiscovery/d3m/issues/380)
  [!304](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/304)
* Fixed `source.from` metadata in datasets and problem descriptions
  and its validation for metalearning database.
  [#363](https://gitlab.com/datadrivendiscovery/d3m/issues/363)
  [!303](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/303)
* Fixed pipeline run references when running the runtime through
  evaluation command.
  [#395](https://gitlab.com/datadrivendiscovery/d3m/issues/395)
  [!294](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/294)
* The core package scoring primitive has been updated to have digest.
  This allows the core package scoring pipeline to have it as well.
  This changes makes it required for the core package to be installed
  in editable mode (`pip3 install -e ...`) when being installed from the
  git repository.
  [!280](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/280)
  **Backwards incompatible.**  

### Other

* Few top-level runtime functions had some of their arguments moved
  to keyword-only arguments:
    * `fit`: `problem_description`
    * `score`: `scoring_pipeline`, `problem_description`, `metrics`, `predictions_random_seed`
    * `prepare_data`: `data_pipeline`, `problem_description`, `data_params`
    * `evaluate`: `data_pipeline`, `scoring_pipeline`, `problem_description`, `data_params`, `metrics`  
  
    [#352](https://gitlab.com/datadrivendiscovery/d3m/issues/352)
    [!301](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/301)
    **Backwards incompatible.**  

* `can_accept` method has been removed from primitive interfaces.
  [#334](https://gitlab.com/datadrivendiscovery/d3m/issues/334)
  [!300](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/300)
  **Backwards incompatible.**    
* NetworkX objects are not anymore container types and are not allowed
  anymore to be passed as values between primitives. Dataset loader now
  does not convert a GML file to a NetworkX object but represents it
  as a files collection resource. A primitive should then convert that
  resource into a normalized edge-list graph representation.
  [#349](https://gitlab.com/datadrivendiscovery/d3m/issues/349)
  [!299](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/299)
  **Backwards incompatible.**  
* `JACCARD_SIMILARITY_SCORE` metric is now a binary metric and requires
  `pos_label` parameter.
  [!299](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/299)
  **Backwards incompatible.**  
* Updated core dependencies. Some important packages are now at versions:
    * `tensorflow`: 2.0.0
    * `keras`: 2.3.1
    * `torch`: 1.3.0.post2
    * `theano`: 1.0.4
    * `scikit-learn`: 0.21.3
    * `numpy`: 1.17.3
    * `pandas`: 0.25.2
    * `networkx`: 2.4
    * `pyarrow`: 0.15.1
    * `scipy`: 1.3.1

    [#398](https://gitlab.com/datadrivendiscovery/d3m/issues/398)
    [#379](https://gitlab.com/datadrivendiscovery/d3m/issues/379)
    [!299](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/299)

* Primitive family `DIMENSIONALITY_REDUCTION` has been added.
  [!284](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/284)
* Added to `algorithm_types`:
    * `POLYNOMIAL_REGRESSION`
    * `IMAGENET`
    * `RETINANET`

    [!306](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/306)

* `--process-dependency-link` is not anymore suggested to be used when
  installing primitives.
* `sample_rate` metadata field inside `dimension` has been renamed to
  `sampling_rate` to make it consistent across metadata. This field
  should contain a sampling rate used for the described dimension,
  when values in the dimension are sampled.
  **Backwards incompatible.**  

## v2019.6.7

### Enhancements

* Dataset loading has been optimized for the case when only one file
  type exists in a file collection. Metadata is also simplified in this case.
  [#314](https://gitlab.com/datadrivendiscovery/d3m/issues/314)
  [!277](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/277)
* Support defining unfitted primitives in the pipeline for passing them
  to another primitive as a hyper-parameter. Unfitted primitives do not
  have any input connected and runtime just creates a primitive instance
  but does not fit or produce them. It then passes this primitive instance
  to another primitive as a hyper-parameter value.
  [!274](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/274)
* When saving datasets, we now use hard-linking of files when possible.
  [#368](https://gitlab.com/datadrivendiscovery/d3m/issues/368)
  [!271](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/271)

### Bugfixes

* Specifying `-E` to the `d3m runtime` CLI now exposes really all outputs
  of all steps and not just pipeline outputs.
  [#367](https://gitlab.com/datadrivendiscovery/d3m/issues/367)
  [!270](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/270)
* Fixed minor issues when loading sklearn example datasets.
* Fixed PyPi metadata of the package.
  [!267](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/267)
* When saving D3M dataset, also structural type information is now used to set
  column type.
  [#339](https://gitlab.com/datadrivendiscovery/d3m/issues/339)
  [!255](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/255)
* When saving D3M dataset, update digest of saved dataset to digest of
  what has been saved.
  [#340](https://gitlab.com/datadrivendiscovery/d3m/issues/340)
  [!262](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/262)

### Other

* Pipeline's `get_exposable_outputs` method has been renamed to `get_producing_outputs`.
  [!270](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/270)
* Updating columns from DataFrame returned from `DataFrame.select_columns`
  does not raise a warning anymore.
  [!268](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/268)
* Added `scipy==1.2.1` as core dependency.
  [!266](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/266)
* Added code style guide to the repository.
  [!260](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/260)
* Added to `algorithm_types`:

    * `ITERATIVE_LABELING`

    [!276](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/276)

## v2019.5.8

* This release contains an implementation of `D3MDatasetSaver` so `Dataset` objects
  can now be saved using their `save` method into D3M dataset format.
* Additional hyper-parameters classes have been defined and existing improved.
  Probably the most useful addition is `List` hyper-parameter which allows
  repeated values with order of values (in contrast with `Set`).
* Standard graph representation has been standardized (a nodelist table and an
  edge list table) and related semantic types have been added to mark source
  and target columns for edges.
* Standard time-series representation has been standardized (a long format)
  and related semantic types have been added to identify columns to index
  time-series by.
* Feature construction primitive should mark newly constructed attributes
  with `https://metadata.datadrivendiscovery.org/types/ConstructedAttribute`
  semantic type.
* There are now mixins available to define primitives which can be used to
  describe neural networks as pipelines.
* There is now a single command line interface for the core package under
  `python3 -m d3m`.

### Enhancements

* Runtime now raises an exception if target columns from problem description
  could not be found in provided input datasets.
  [#281](https://gitlab.com/datadrivendiscovery/d3m/issues/281)
  [!155](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/155)
* Core package command line interfaces have been consolidated and revamped
  and are now all available under single `python3 -m d3m`.
  [#338](https://gitlab.com/datadrivendiscovery/d3m/issues/338)
  [!193](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/193)
  [!233](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/233)
* Added `--expose-produced-outputs` argument runtime CLI to allow saving
  to a directory produced outputs of all primitives from pipeline's run.
  Useful for debugging.
  [#206](https://gitlab.com/datadrivendiscovery/d3m/issues/206)
  [!223](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/223)
* CSVLoader and SklearnExampleLoader dataset loaders now add
  `d3mIndex` column if one does not exist already.
  [#266](https://gitlab.com/datadrivendiscovery/d3m/issues/266)
  [!202](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/202)
* Added `--not-standard-pipeline` argument to `fit`, `produce`, and `fit-produce`
  runtime CLI to allow running non-standard pipelines.
  [#312](https://gitlab.com/datadrivendiscovery/d3m/issues/312)
  [!228](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/228)
* Sampling `Bounded` and base `Hyperparameter` hyper-parameter now issues
  a warning that sampling of those hyper-parameters is ill-defined.
  [!220](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/220)
* `Bounded` hyper-parameter with both bounds now samples from uniform
  distribution.
  [!220](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/220)
* Added new hyper-parameter classes: `SortedSet`, `List`, and `SortedList`.
  [#236](https://gitlab.com/datadrivendiscovery/d3m/issues/236)
  [#292](https://gitlab.com/datadrivendiscovery/d3m/issues/292)
  [!219](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/219)
* All bounded hyper-parameter classes now accept additional arguments to
  control if bounds are inclusive or exclusive.
  [#199](https://gitlab.com/datadrivendiscovery/d3m/issues/199)
  [!215](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/215)
* `Dataset` objects can now be saved to D3M dataset format by
  calling `save` method on them.
  [#31](https://gitlab.com/datadrivendiscovery/d3m/issues/31)
  [#344](https://gitlab.com/datadrivendiscovery/d3m/issues/344)
  [!96](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/96)
  [!217](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/217)

### Bugfixes

* Fixed `NormalizeMutualInformationMetric` implementation. 
  [#357](https://gitlab.com/datadrivendiscovery/d3m/issues/357)
  [!257](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/257)
* JSON representation of `Union` hyper-parameter values and other
  pickled hyper-parameter values has been changed to assure better
  interoperability.
  [#359](https://gitlab.com/datadrivendiscovery/d3m/issues/359)
  [!256](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/256)
  **Backwards incompatible.**  
* All d3m schemas are now fully valid according to JSON schema draft v4.
  [#79](https://gitlab.com/datadrivendiscovery/d3m/issues/79)
  [!233](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/233)
* Fixed an error when saving a fitted pipeline to stdout.
  [#353](https://gitlab.com/datadrivendiscovery/d3m/issues/353)
  [!250](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/250)
* Hyper-parameters cannot use `NaN` and infinity floating-point values
  as their bounds. This assures compatibility with JSON.
  [#324](https://gitlab.com/datadrivendiscovery/d3m/issues/324)
  [!237](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/237)
  **Backwards incompatible.**  
* Pipelines are now exported to JSON in strict compliance of the
  JSON specification.
  [#323](https://gitlab.com/datadrivendiscovery/d3m/issues/323)
  [!238](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/238)
* Runtime execution does not fail anymore if predictions cannot be converted
  to JSON for pipeline run. A warning is issued instead.
  [#347](https://gitlab.com/datadrivendiscovery/d3m/issues/347)
  [!227](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/227)
* Better support for running reference runtime without exceptions on non-Linux
  operating systems.
  [#246](https://gitlab.com/datadrivendiscovery/d3m/issues/246)
  [!218](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/218)
* Strict checking of dataset, pipeline and primitive digests against those provided
  in metadata are now correctly controlled using `--strict-digest`/`strict_digest`
  arguments.
  [#346](https://gitlab.com/datadrivendiscovery/d3m/issues/346)
  [!213](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/213)
* Fixed error propagation in `evaluate` runtime function, if error
  happens during scoring.
  [!210](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/210)
* Fixed accessing container DataFrame's `metadata` attribute when
  DataFrame also contains a column with the name `metadata`.
  [#330](https://gitlab.com/datadrivendiscovery/d3m/issues/330)
  [!201](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/201)
* Fixed `.meta` file resolving when `--datasets` runtime argument
  is not an absolute path.
  [!194](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/194)
* Fixed `get_relations_graph` resolving of column names (used in `Denormalize`
  common primitive).
  [!196](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/196)

### Other

* Other validation functions for metalearning documents. This includes
  also CLI to validate.
  [#220](https://gitlab.com/datadrivendiscovery/d3m/issues/220)
  [!233](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/233)
* Pipeline run schema now requires scoring dataset inputs to be recorded
  if a data preparation pipeline has not been used.
  [!243](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/243)
  **Backwards incompatible.**  
* Core package now provides standard scoring primitive and scoring pipeline
  which are used by runtime by default.
  [#307](https://gitlab.com/datadrivendiscovery/d3m/issues/307)
  [!231](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/231)
* Pipeline run can now be generated also for a subset of non-standard
  pipelines: those which have all inputs of `Dataset` type.
  [!232](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/232)
* Pipeline run now also records a normalized score, if available.
  [!230](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/230)
* Pipeline `context` field has been removed from schema and implementation.
  [!229](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/229)
* Added `pure_primitive` field to primitive's metadata so that primitives
  can mark themselves as not pure (by default all primitives are seen as pure).
  [#331](https://gitlab.com/datadrivendiscovery/d3m/issues/331)
  [!226](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/226)
* `Metadata` methods `to_json_structure` and `to_simple_structure` has been
  modified to not return anymore internal metadata representation but
  metadata representation equivalent to what you get from `query` call.
  To obtain internal representation use `to_internal_json_structure`
  and `to_internal_simple_structure`.
  [!225](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/225)
  **Backwards incompatible.**  
* `NeuralNetworkModuleMixin` and `NeuralNetworkObjectMixin` have been
  added to primitive interfaces to support representing neural networks
  as pipelines.
  [#174](https://gitlab.com/datadrivendiscovery/d3m/issues/174)
  [!87](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/87)
* `get_loss_function` has been renamed to `get_loss_metric` in
  `LossFunctionMixin`.
  [!87](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/87)
  **Backwards incompatible.**  
* `UniformInt`, `Uniform`, and `LogUniform` hyper-parameter classes now
  subclass `Bounded` class. 
  [!216](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/216)
* Metrics do not have default parameter values anymore, cleaned legacy
  parts of code assuming so.
  [!212](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/212)
* Added new semantic types:
  * `https://metadata.datadrivendiscovery.org/types/EdgeSource`
  * `https://metadata.datadrivendiscovery.org/types/DirectedEdgeSource`
  * `https://metadata.datadrivendiscovery.org/types/UndirectedEdgeSource`
  * `https://metadata.datadrivendiscovery.org/types/SimpleEdgeSource`
  * `https://metadata.datadrivendiscovery.org/types/MultiEdgeSource`
  * `https://metadata.datadrivendiscovery.org/types/EdgeTarget`
  * `https://metadata.datadrivendiscovery.org/types/DirectedEdgeTarget`
  * `https://metadata.datadrivendiscovery.org/types/UndirectedEdgeTarget`
  * `https://metadata.datadrivendiscovery.org/types/SimpleEdgeTarget`
  * `https://metadata.datadrivendiscovery.org/types/MultiEdgeTarget`
  * `https://metadata.datadrivendiscovery.org/types/ConstructedAttribute`
  * `https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey`
  * `https://metadata.datadrivendiscovery.org/types/GroupingKey`

    [#134](https://gitlab.com/datadrivendiscovery/d3m/issues/134)
    [#348](https://gitlab.com/datadrivendiscovery/d3m/issues/348)
    [!211](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/211)
    [!214](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/214)

* Updated core dependencies. Some important packages are now at versions:
    * `scikit-learn`: 0.20.3
    * `pyarrow`: 0.13.0

    [!206](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/206)

* Clarified in primitive interface documentation that if primitive should have been
  fitted before calling its produce method, but it has not been, primitive should
  raise a ``PrimitiveNotFittedError`` exception.
  [!204](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/204)
* Added to `algorithm_types`:

    * `EQUI_JOIN`
    * `DATA_RETRIEVAL`
    * `DATA_MAPPING`
    * `MAP`
    * `INFORMATION_THEORETIC_METAFEATURE_EXTRACTION`
    * `LANDMARKING_METAFEATURE_EXTRACTION`
    * `MODEL_BASED_METAFEATURE_EXTRACTION`
    * `STATISTICAL_METAFEATURE_EXTRACTION`
    * `VECTORIZATION`
    * `BERT`

    [!160](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/160)
    [!186](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/186)
    [!224](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/224)
    [!247](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/247)

* Primitive family `METAFEATURE_EXTRACTION` has been renamed to `METALEARNING`.
  [!160](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/160)
  **Backwards incompatible.**  

## v2019.4.4

* With this release metadata is not automatically generated anymore when DataFrame or ndarray
  is being wrapped into a corresponding container type. Now you have to explicitly set
  `generate_metadata` constructor argument to `True` or call `generate` method on metadata
  object afterwards.
  This has been changed to improve performance of many primitives and operations on
  container types which were slowed down because of unnecessary and unexpected
  generation of metadata.
  This change requires manual inspection of primitive's code to determine what change
  is necessary. Some suggestions what to look for:
    * `set_for_value` method has been deprecated: generally it can be replaced with `generate`
      call, or even removed in some cases:
        * `value.metadata = value.metadata.set_for_value(value, generate_metadata=False)` remove.
        * `value.metadata = new_metadata.set_for_value(value, generate_metadata=False)` replace with `value.metadata = new_metadata`.
        * `value.metadata = new_metadata.set_for_value(value, generate_metadata=True)` replace with `value.metadata = new_metadata.generate(value)`.
    * `clear` method has been deprecated: generally you can now instead simply create
      a fresh instance of `DataMetadata`, potentially calling `generate` as well:
        * `outputs_metadata = inputs_metadata.clear(new_metadata, for_value=outputs, generate_metadata=True)` replace with
          `outputs_metadata = metadata_base.DataMetadata(metadata).generate(outputs)`.
        * `outputs_metadata = inputs_metadata.clear(for_value=outputs, generate_metadata=False)` replace with
          `outputs_metadata = metadata_base.DataMetadata()`.
    * Search for all calls to constructors of `container.List`, `container.ndarray`,
      `container.Dataset`, `container.DataFrame` container types and explicitly set
      `generate_metadata` to `True`. Alternatively, you can also manually update
      metadata instead of relying on automatic metadata generation.
    * The main idea is that if you are using automatic metadata generation in your primitive,
      make sure you generate it only once, just before you return container type from
      your primitive. Of course, if you call code which expects metadata from inside your primitive,
      you might have to assure or generate metadata before calling that code as well.

### Enhancements

* Primitives now get a `temporary_directory` constructor argument pointing
  to a directory they can use to store any files for the duration of current pipeline
  run phase. The main intent of this temporary directory is to store files referenced
  by any ``Dataset`` object your primitive might create and followup primitives in
  the pipeline should have access to. To support configuration of the location of these
  temporary directories, the reference runtime now has a `--scratch` command line argument
  and corresponding `scratch_dir` constructor argument. 
  [#306](https://gitlab.com/datadrivendiscovery/d3m/issues/306)
  [!190](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/190)
* Made sure that number of inputs provided to the runtime has to match the number of inputs a pipeline accepts.
  [#301](https://gitlab.com/datadrivendiscovery/d3m/issues/301)
  [!183](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/183)
* Supported MIT-LL dataset and problem schemas version 3.3.0. Now all suggested targets and suggested privileged data
  columns are now by default also attributes. Runtime makes sure that if any column is marked as problem description's
  target it is not marked as an attribute anymore.
  [#291](https://gitlab.com/datadrivendiscovery/d3m/issues/291)
  [!182](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/182)
  **Backwards incompatible.**
* `steps` and `method_calls` made optional in pipeline run schema to allow easier recording of failed pipelines.
  [!167](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/167)
* Pipeline run now records also start and end timestamps of pipelines and steps.
  [#258](https://gitlab.com/datadrivendiscovery/d3m/issues/258)
  [!162](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/162)
* `Metadata` has two new methods to query metadata, `query_field` and `query_field_with_exceptions`
  which you can use when you want to query just a field of metadata, and not whole metadata object.
  Similarly, `DataMetadata` has a new method `query_column_field`.
* `DataMetadata`'s `generate` method has now `compact` argument to control it automatically
  generated metadata is compacted (if all elements of a dimension have equal metadata, it is
  compacted into `ALL_ELEMENTS` selector segment) or not (default).
  There is also a `compact` method available on `Metadata` to compact metadata on demand.
* Automatically generated metadata is not automatically compacted anymore by default
  (compacting is when all elements of a dimension have equal metadata, moving that
  metadata `ALL_ELEMENTS` selector segment).
* `generate_metadata` argument of container types' constructors has been switched
  from default `True` to default `False` to prevent unnecessary and unexpected
  generation of metadata, slowing down execution of primitives. Moreover,
  `DataMetadata` has now a method `generate` which can be used to explicitly
  generate and update metadata given a data value.
  Metadata methods `set_for_value` and `clear` have been deprecated and can
  be generally replaced with `generate` call, or creating a new metadata
  object, or removing the call. 
  **Backwards incompatible.**
  [#143](https://gitlab.com/datadrivendiscovery/d3m/issues/143)
  [!180](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/180)
* Loading of datasets with many files has been heavily optimized.
  [#164](https://gitlab.com/datadrivendiscovery/d3m/issues/164)
  [!136](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/136)
  [!178](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/178)
  [!](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/179)
* Extended container's `DataFrame.to_csv` method to use by default
  metadata column names for CSV header instead of column names of
  `DataFrame` itself.
  [!158](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/158).
* Problem parsing has been refactored into extendable system similar to how
  dataset parsing is done. A simple `d3m.metadata.problem.Problem` class has
  been defined to contain a problem description. Default implementation supports
  loading of D3M problems. `--problem` command line argument to reference runtime
  can now be a path or URI to a problem description.
  [#276](https://gitlab.com/datadrivendiscovery/d3m/issues/276)
  [!145](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/145)
* Data metadata is not validated anymore at every update, but only when explicitly
  validated using the `check` method. This improves metadata performance.
  [!144](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/144)

### Other

* Top-level runtime functions now also return `Result` (or new `MultiResult`)
  objects instead of raising special `PipelineRunError` exception (which has been
  removed) and instead of returning just pipeline run (which is available
  inside `Result`).
  [#297](https://gitlab.com/datadrivendiscovery/d3m/issues/297)
  [!192](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/192)
  **Backwards incompatible.**
* Metrics have been reimplemented to operate on whole predictions DataFrame.
  [#304](https://gitlab.com/datadrivendiscovery/d3m/issues/304)
  [#311](https://gitlab.com/datadrivendiscovery/d3m/issues/311)
  [!171](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/171)
  **Backwards incompatible.**
* Pipeline run implementation has been refactored to be in a single class to
  facilitate easier subclassing.
  [#255](https://gitlab.com/datadrivendiscovery/d3m/issues/255)
  [#305](https://gitlab.com/datadrivendiscovery/d3m/issues/305)
  [!164](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/164)
* Added new semantic types:
  * `https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey`
  * `https://metadata.datadrivendiscovery.org/types/BoundingPolygon`
  * `https://metadata.datadrivendiscovery.org/types/UnknownType`
* Removed semantic types:
    * `https://metadata.datadrivendiscovery.org/types/BoundingBox`
    * `https://metadata.datadrivendiscovery.org/types/BoundingBoxXMin`
    * `https://metadata.datadrivendiscovery.org/types/BoundingBoxYMin`
    * `https://metadata.datadrivendiscovery.org/types/BoundingBoxXMax`
    * `https://metadata.datadrivendiscovery.org/types/BoundingBoxYMax`

    **Backwards incompatible.**
* Added to `primitive_family`:
  * `SEMISUPERVISED_CLASSIFICATION`
  * `SEMISUPERVISED_REGRESSION`
  * `VERTEX_CLASSIFICATION`
* Added to `task_type`:
  * `SEMISUPERVISED_CLASSIFICATION`
  * `SEMISUPERVISED_REGRESSION`
  * `VERTEX_CLASSIFICATION`
* Added to `performance_metric`:
  * `HAMMING_LOSS`
* Removed from `performance_metric`:
    * `ROOT_MEAN_SQUARED_ERROR_AVG`
    
    **Backwards incompatible.**
* Added `https://metadata.datadrivendiscovery.org/types/GPUResourcesUseParameter` and
  `https://metadata.datadrivendiscovery.org/types/CPUResourcesUseParameter` semantic types for
  primitive hyper-parameters which control the use of GPUs and CPUs (cores), respectively.
  You can use these semantic types to mark which hyper-parameter defines a range of how many
  GPUs or CPUs (cores), respectively, a primitive can and should use.
  [#39](https://gitlab.com/datadrivendiscovery/d3m/issues/39)
  [!177](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/177)
* Added `get_hyperparams` and `get_volumes` helper methods to `PrimitiveMetadata`
  so that it is easier to obtain hyper-parameters definitions class of a primitive.
  [#163](https://gitlab.com/datadrivendiscovery/d3m/issues/163)
  [!175](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/175)
* Pipeline run schema now records the global seed used by the runtime to run the pipeline.
  [!187](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/187)
* Core package scores output now includes also a random seed column.
  [#299](https://gitlab.com/datadrivendiscovery/d3m/issues/299)
  [!185](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/185)
* Metrics in core packages now take as input whole predictions DataFrame
  objects and compute scores over them. So `applicability_to_targets` metric
  method has been removed, and also code which handles the list of target
  columns metric used to compute the score. This is not needed anymore
  because now all columns are always used by all metrics. Moreover,
  corresponding `dataset_id` and `targets` fields have been removed from
  pipeline run schema.
* Core package now requires pip 19 or later to be installed.
  `--process-dependency-links` argument when installing the package is not needed
  nor supported anymore.
  Primitives should not require use of `--process-dependency-links` to install
  them either. Instead use link dependencies as described in
  [PEP 508](https://www.python.org/dev/peps/pep-0508/).
  [#285](https://gitlab.com/datadrivendiscovery/d3m/issues/285)
  [!176](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/176)
  **Backwards incompatible.**
* `outputs` field in parsed problem description has been removed.
  [#290](https://gitlab.com/datadrivendiscovery/d3m/issues/290)
  [!174](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/174)
  **Backwards incompatible.**
* `Hyperparameter`'s `value_to_json` and `value_from_json` methods have been
  renamed to `value_to_json_structure` and `value_from_json_structure`, respectively.
  [#122](https://gitlab.com/datadrivendiscovery/d3m/issues/122)
  [#173](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/173)
* Moved utility functions from common primitives package to core package:

    * `copy_metadata` to `Metadata.copy_to` method
    * `select_columns` to `DataFrame.select_columns` method
    * `select_columns_metadata` to `DataMetadata.select_columns` method
    * `list_columns_with_semantic_types` to `DataMetadata.list_columns_with_semantic_types` method
    * `list_columns_with_structural_types` to `DataMetadata.list_columns_with_structural_types` method
    * `remove_columns` to `DataFrame.remove_columns` method
    * `remove_columns_metadata` to `DataMetadata.remove_columns` method
    * `append_columns` to `DataFrame.append_columns` method
    * `append_columns_metadata` to `DataMetadata.append_columns` method
    * `insert_columns` to `DataFrame.insert_columns` method
    * `insert_columns_metadata` to `DataMetadata.insert_columns` method
    * `replace_columns` to `DataFrame.replace_columns` method
    * `replace_columns_metadata` to `DataMetadata.replace_columns` method
    * `get_index_columns` to `DataMetadata.get_index_columns` method
    * `horizontal_concat` to `DataFrame.horizontal_concat` method
    * `horizontal_concat_metadata` to `DataMetadata.horizontal_concat` method
    * `get_columns_to_use` to `d3m.base.utils.get_columns_to_use` function
    * `combine_columns` to `d3m.base.utils.combine_columns` function
    * `combine_columns_metadata` to `d3m.base.utils.combine_columns_metadata` function
    * `set_table_metadata` to `DataMetadata.set_table_metadata` method
    * `get_column_index_from_column_name` to `DataMetadata.get_column_index_from_column_name` method
    * `build_relation_graph` to `Dataset.get_relations_graph` method
    * `get_tabular_resource` to `d3m.base.utils.get_tabular_resource` function
    * `get_tabular_resource_metadata` to `d3m.base.utils.get_tabular_resource_metadata` function
    * `cut_dataset` to `Dataset.select_rows` method

    [#148](https://gitlab.com/datadrivendiscovery/d3m/issues/148)
    [!172](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/172)

* Updated core dependencies. Some important packages are now at versions:
    * `pyarrow`: 0.12.1

    [!156](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/156)

## v2019.2.18

### Bugfixes

* JSON schema for problem descriptions has been fixed to allow loading
  D3M problem descriptions with data augmentation fields.
  [#284](https://gitlab.com/datadrivendiscovery/d3m/issues/284)
  [!154](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/154)
* Utils now contains representers to encode numpy float and integer numbers
  for YAML. Importing `utils` registers them.
  [#275](https://gitlab.com/datadrivendiscovery/d3m/issues/275)
  [!148](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/148)
* Made sure all JSON files are read with UTF-8 encoding, so that we do
  not depend on the encoding of the environment.
  [!150](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/150)
  [!153](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/153)

## v2019.2.12

### Enhancements

* Runtime now makes sure that target columns are never marked as attributes.
  [#265](https://gitlab.com/datadrivendiscovery/d3m/issues/265)
  [!131](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/131)
* When using runtime CLI, pipeline run output is made even in the case of an
  exception. Moreover, exception thrown from `Result.check_success` contains
  associated pipeline runs in its `pipeline_runs` attribute.
  [#245](https://gitlab.com/datadrivendiscovery/d3m/issues/245)
  [!120](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/120)
* Made additional relaxations when reading D3M datasets and problem descriptions
  to not require required fields which have defaults.
  [!128](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/128)
* When loading D3M datasets and problem descriptions, package now just warns
  if they have an unsupported schema version and continues to load them.
  [#247](https://gitlab.com/datadrivendiscovery/d3m/issues/247)
  [!119](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/119)
* Added to `primitive_family`:

    * `NATURAL_LANGUAGE_PROCESSING`

    [!125](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/125)

### Bugfixes

* Fixed an unexpected exception when running a pipeline using reference
  runtime but not requesting to return output values.
  [#260](https://gitlab.com/datadrivendiscovery/d3m/issues/260)
  [!127](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/127)
* Fixed infinite recursion loop which happened if Python logging was
  configured inside primitive's method call. Moreover, recording of
  logging records for pipeline run changed so that it does not modify
  the record itself while recording it.
  [#250](https://gitlab.com/datadrivendiscovery/d3m/issues/250)
  [#123](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/123)
* Correctly populate `volumes` primitive constructor argument.
  Before it was not really possible to use primitive static files with
  reference runtime.
  [!132](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/132)
* Fixed runtime/pipeline run configuration through environment variables.
  Now it reads them without throwing an exception.
  [#274](https://gitlab.com/datadrivendiscovery/d3m/issues/274)
  [!118](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/118)
  [!137](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/137)

## v2019.1.21

* Some enumeration classes were moved and renamed:
    * `d3m.metadata.pipeline.ArgumentType` to `d3m.metadata.base.ArgumentType`
    * `d3m.metadata.pipeline.PipelineContext` to `d3m.metadata.base.Context`
    * `d3m.metadata.pipeline.PipelineStep` to `d3m.metadata.base.PipelineStepType`

    **Backwards incompatible.**

* Added `pipeline_run.json` JSON schema which describes the results of running a
  pipeline as described by the `pipeline.json` JSON schema. Also implemented
  a reference pipeline run output for reference runtime.
  [#165](https://gitlab.com/datadrivendiscovery/d3m/issues/165)
  [!59](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/59)
* When computing primitive digests, primitive's ID is included in the
  hash so that digest is not the same for all primitives from the same
  package.
  [#154](https://gitlab.com/datadrivendiscovery/d3m/issues/154)
* When datasets are loaded, digest of their metadata and data can be
  computed. To control when this is done, `compute_digest` argument
  to `Dataset.load` can now take the following `ComputeDigest`
  enumeration values: `ALWAYS`, `ONLY_IF_MISSING` (default), and `NEVER`.
  [!75](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/75)
* Added `digest` field to pipeline descriptions. Digest is computed based
  on the pipeline document and it helps differentiate between pipelines
  with same `id`. When loading a pipeline, if there
  is a digest mismatch a warning is issued. You can use
  `strict_digest` argument to request an exception instead.
  [#190](https://gitlab.com/datadrivendiscovery/d3m/issues/190)
  [!75](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/75)
* Added `digest` field to problem description metadata.
  This `digest` field is computed based on the problem description document
  and it helps differentiate between problem descriptions with same `id`.
  [#190](https://gitlab.com/datadrivendiscovery/d3m/issues/190)
  [!75](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/75)
* Moved `id`, `version`, `name`, `other_names`, and `description` fields
  in problem schema to top-level of the problem description. Moreover, made
  `id` required. This aligns it more with the structure of other descriptions we have.
  [!75](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/75)
  **Backwards incompatible.**
* Pipelines can now provide multiple inputs to the same primitive argument.
  In such case runtime wraps those inputs into a `List` container type, and then
  passes the list to the primitive.
  [#200](https://gitlab.com/datadrivendiscovery/d3m/issues/200)
  [!112](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/112)
* Primitives now have a method `fit_multi_produce` which primitive author can
  override to implement an optimized version of both fitting and producing a primitive on same data.
  The default implementation just calls `set_training_data`, `fit` and produce methods.
  If your primitive has non-standard additional arguments in its `produce` method(s) then you
  will have to implement `fit_multi_produce` method to accept those additional arguments
  as well, similarly to how you have had to do for `multi_produce`.
  [#117](https://gitlab.com/datadrivendiscovery/d3m/issues/117)
  [!110](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/110)
  **Could be backwards incompatible.**
* `source`, `timestamp`, and `check` arguments to all metadata functions and container types'
  constructors have been deprecated. You do not have to and should not be providing them anymore.
  [#171](https://gitlab.com/datadrivendiscovery/d3m/issues/171)
  [#172](https://gitlab.com/datadrivendiscovery/d3m/issues/172)
  [#173](https://gitlab.com/datadrivendiscovery/d3m/issues/173)
  [!108](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/108)
  [!109](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/109)
* Primitive's constructor is not run anymore during importing of primitive's class
  which allows one to use constructor to load things and do any resource
  allocation/reservation. Constructor is now the preferred place to do so.
  [#158](https://gitlab.com/datadrivendiscovery/d3m/issues/158)
  [!107](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/107)
* `foreign_key` metadata has been extended with `RESOURCE` type which allows
  referencing another resource in the same dataset.
  [#221](https://gitlab.com/datadrivendiscovery/d3m/issues/221)
  [!105](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/105)
* Updated supported D3M dataset and problem schema both to version 3.2.0.
  Problem description parsing supports data augmentation metadata.
  A new approach for LUPI datasets and problems is now supported,
  including runtime support.
  Moreover, if dataset's resource name is `learningData`, it is marked as a
  dataset entry point.
  [#229](https://gitlab.com/datadrivendiscovery/d3m/issues/229)
  [#225](https://gitlab.com/datadrivendiscovery/d3m/issues/225)
  [#226](https://gitlab.com/datadrivendiscovery/d3m/issues/226)
  [!97](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/97)
* Added support for "raw" datasets.
  [#217](https://gitlab.com/datadrivendiscovery/d3m/issues/217)
  [!94](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/94)
* A warning is issued if a primitive does not provide a description through
  its docstring.
  [#167](https://gitlab.com/datadrivendiscovery/d3m/issues/167)
  [!101](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/101)
* A warning is now issued if an installable primitive is lacking contact or bug
  tracker URI metadata.
  [#178](https://gitlab.com/datadrivendiscovery/d3m/issues/178)
  [!81](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/81)
* `Pipeline` class now has also `equals` and `hash` methods which can help
  determining if two pipelines are equal in the sense of isomorphism.
  [!53](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/53)
* `Pipeline` and pipeline steps classes now has `get_all_hyperparams`
  method to return all hyper-parameters defined for a pipeline and steps.
  [#222](https://gitlab.com/datadrivendiscovery/d3m/issues/222)
  [!104](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/104)
* Implemented a check for primitive Python paths to assure that they adhere
  to the new standard of all of them having to be in the form `d3m.primitives.primitive_family.primitive_name.kind`
  (e.g., `d3m.primitives.classification.random_forest.SKLearn`).
  Currently there is a warning if a primitive has a different Python path,
  and after January 2019 it will be an error.
  For `primitive_name` segment there is a [`primitive_names.py`](./d3m/metadata/primitive_names.py)
  file containing a list of all allowed primitive names.
  Everyone is encouraged to help currate this list and suggest improvements (merging, removals, additions)
  of values in that list. Initial version was mostly automatically made from an existing list of
  values used by current primitives.
  [#3](https://gitlab.com/datadrivendiscovery/d3m/issues/3)
  [!67](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/67)
* Added to semantic types:
   * `https://metadata.datadrivendiscovery.org/types/TokenizableIntoNumericAndAlphaTokens`
   * `https://metadata.datadrivendiscovery.org/types/TokenizableByPunctuation`
   * `https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber`
   * `https://metadata.datadrivendiscovery.org/types/UnspecifiedStructure`
   * `http://schema.org/email`
   * `http://schema.org/URL`
   * `http://schema.org/address`
   * `http://schema.org/State`
   * `http://schema.org/City`
   * `http://schema.org/Country`
   * `http://schema.org/addressCountry`
   * `http://schema.org/postalCode`
   * `http://schema.org/latitude`
   * `http://schema.org/longitude`

   [!62](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/62)
   [!95](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/95)
   [!94](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/94)

* Updated core dependencies. Some important packages are now at versions:
    * `scikit-learn`: 0.20.2
    * `numpy`: 1.15.4
    * `pandas`: 0.23.4
    * `networkx`: 2.2
    * `pyarrow`: 0.11.1

    [#106](https://gitlab.com/datadrivendiscovery/d3m/issues/106)
    [#175](https://gitlab.com/datadrivendiscovery/d3m/issues/175)

* Added to `algorithm_types`:
  * `IDENTITY_FUNCTION`
  * `DATA_SPLITTING`
  * `BREADTH_FIRST_SEARCH`
* Moved a major part of README to Sphinx documentation which is built
  and available at [http://docs.datadrivendiscovery.org/](http://docs.datadrivendiscovery.org/).
* Added a `produce_methods` argument to `Primitive` hyper-parameter class
  which allows one to limit matching primitives only to those providing all
  of the listed produce methods.
  [#124](https://gitlab.com/datadrivendiscovery/d3m/issues/124)
  [!56](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/56)
* Fixed `sample_multiple` method of the `Hyperparameter` class.
  [#157](https://gitlab.com/datadrivendiscovery/d3m/issues/157)
  [!50](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/50)
* Fixed pickling of `Choice` hyper-parameter.
  [!49](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/49)
  [!51](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/51)
* Added `Constant` hyper-parameter class.
  [#186](https://gitlab.com/datadrivendiscovery/d3m/issues/186)
  [!90](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/90)
* Added `count` to aggregate values in metafeatures.
  [!52](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/52)
* Clarified and generalized some metafeatures, mostly renamed so that it can be
  used on attributes as well:
    * `number_of_classes` to `number_distinct_values`
    * `class_entropy` to `entropy_of_values`
    * `majority_class_ratio` to `value_probabilities_aggregate.max`
    * `minority_class_ratio` to `value_probabilities_aggregate.min`
    * `majority_class_size` to `value_counts_aggregate.max`
    * `minority_class_size` to `value_counts_aggregate.min`
    * `class_probabilities` to `value_probabilities_aggregate`
    * `target_values` to `values_aggregate`
    * `means_of_attributes` to `mean_of_attributes`
    * `standard_deviations_of_attributes` to `standard_deviation_of_attributes`
    * `categorical_joint_entropy` to `joint_entropy_of_categorical_attributes`
    * `numeric_joint_entropy` to `joint_entropy_of_numeric_attributes`
    * `pearson_correlation_of_attributes` to `pearson_correlation_of_numeric_attributes`
    * `spearman_correlation_of_attributes` to `spearman_correlation_of_numeric_attributes`
    * `canonical_correlation` to `canonical_correlation_of_numeric_attributes`

    [!52](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/52)

* Added metafeatures:
    * `default_accuracy`
    * `oner`
    * `jrip`
    * `naive_bayes_tree`
    * `number_of_string_attributes`
    * `ratio_of_string_attributes`
    * `number_of_other_attributes`
    * `ratio_of_other_attributes`
    * `attribute_counts_by_structural_type`
    * `attribute_ratios_by_structural_type`
    * `attribute_counts_by_semantic_type`
    * `attribute_ratios_by_semantic_type`
    * `value_counts_aggregate`
    * `number_distinct_values_of_discrete_attributes`
    * `entropy_of_discrete_attributes`
    * `joint_entropy_of_discrete_attributes`
    * `joint_entropy_of_attributes`
    * `mutual_information_of_discrete_attributes`
    * `equivalent_number_of_discrete_attributes`
    * `discrete_noise_to_signal_ratio`

    [!21](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/21)
    [!52](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/52)

* Added special handling when reading scoring D3M datasets (those with true targets in a separate
  file `targets.csv`). When such dataset is detected, the values from the separate file are now
  merged into the dataset, and its ID is changed to finish with `SCORE` suffix. Similarly, an
  ID of a scoring problem description gets its suffix changed to `SCORE`.
  [#176](https://gitlab.com/datadrivendiscovery/d3m/issues/176)
* Organized semantic types and add to some of them parent semantic types to organize/structure
  them better. New parent semantic types added: `https://metadata.datadrivendiscovery.org/types/ColumnRole`,
  `https://metadata.datadrivendiscovery.org/types/DimensionType`, `https://metadata.datadrivendiscovery.org/types/HyperParameter`.
* Fixed that `dateTime` column type is mapped to `http://schema.org/DateTime` semantic
  type and not `https://metadata.datadrivendiscovery.org/types/Time`.
  **Backwards incompatible.**
* Updated generated [site for metadata](https://metadata.datadrivendiscovery.org/) and
  generate sites describing semantic types.
  [#33](https://gitlab.com/datadrivendiscovery/d3m/issues/33)
  [#114](https://gitlab.com/datadrivendiscovery/d3m/issues/114)
  [!37](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/37)
* Optimized resolving of primitives in `Resolver` to not require loading of
  all primitives when loading a pipeline, in the common case.
  [#162](https://gitlab.com/datadrivendiscovery/d3m/issues/162)
  [!38](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/38)
* Added `NotFoundError`, `AlreadyExistsError`, and `PermissionDeniedError`
  exceptions to `d3m.exceptions`.
* `Pipeline`'s `to_json_structure`, `to_json`, and `to_yaml` now have `nest_subpipelines`
  argument which allows conversion with nested sub-pipelines instead of them
  being only referenced.
* Made sure that Arrow serialization of metadata does not pickle also linked
  values (`for_value`).
* Made sure enumerations are picklable.
* `PerformanceMetric` class now has `best_value` and `worst_value` which
  return the range of possible values for the metric. Moreover, `normalize`
  method normalizes the metric's value to a range between 0 and 1.
* Load D3M dataset qualities only after data is loaded. This fixes
  lazy loading of datasets with qualities which was broken before.
* Added `load_all_primitives` argument to the default pipeline `Resolver`
  which allows one to control loading of primitives outside of the resolver.
* Added `primitives_blacklist` argument to the default pipeline `Resolver`
  which allows one to specify a collection of primitive path prefixes to not
  (try to) load.
* Fixed return value of the `fit` method in `TransformerPrimitiveBase`.
  It now correctly returns `CallResult` instead of `None`.
* Fixed a typo and renamed `get_primitive_hyparparams` to `get_primitive_hyperparams`
  in `PrimitiveStep`.
  **Backwards incompatible.**
* Additional methods were added to the `Pipeline` class and step classes,
  to support runtime and easier manipulation of pipelines programmatically
  (`get_free_hyperparams`, `get_input_data_references`, `has_placeholder`,
  `replace_step`, `get_exposable_outputs`).
* Added reference implementation of the runtime. It is available
  in the `d3m.runtime` module. This module also has an extensive
  command line interface you can access through `python3 -m d3m.runtime`.
  [#115](https://gitlab.com/datadrivendiscovery/d3m/issues/115)
  [!57](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/57)
  [!72](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/72)
* `GeneratorPrimitiveBase` interface has been changed so that `produce` method
  accepts a list of non-negative integers as an input instead of a list of `None` values.
  This allows for batching and control by the caller which outputs to generate.
  Previously outputs would depend on number of calls to `produce` and number of outputs
  requested in each call. Now these integers serve as an index into the set of potential
  outputs.
  **Backwards incompatible.**
* We now try to preserve metadata log in default implementation of `can_accept`.
* Added `sample_rate` field to `dimension` metadata.
* `python3 -m d3m.index download` command now accepts `--prefix` argument to limit the
  primitives for which static files are downloaded. Useful for testing.
* Added `check` argument to `DataMetadata`'s `update` and `remove` methods which allows
  one to control if selector check against `for_value` should be done or not. When
  it is known that selector is valid, not doing the check can speed up those methods.
* Defined metadata field `file_columns` which allows to store known columns metadata for
  tables referenced from columns. This is now used by a D3M dataset reader to store known
  columns metadata for collections of CSV files. Previously, this metadata was lost despite
  being available in Lincoln Labs dataset metadata.

## v2018.7.10

* Made sure that `OBJECT_DETECTION_AVERAGE_PRECISION` metric supports operation on
  vectorized target column.
  [#149](https://gitlab.com/datadrivendiscovery/d3m/issues/149)
* Files in D3M dataset collections are now listed recursively to support datasets
  with files split into directories.
  [#146](https://gitlab.com/datadrivendiscovery/d3m/issues/146)
* When parameter value for `Params` fails to type check, a name of the parameter is now
  reported as well.
  [#135](https://gitlab.com/datadrivendiscovery/d3m/issues/135)
* `python3 -m d3m.index` has now additional command `download` which downloads all static
  files needed by available primitives. Those files are then exposed through `volumes`
  constructor argument to primitives by TA2/runtime. Files are stored into an output
  directory in a standard way where each volume is stored with file or directory name
  based on its digest.
  [#102](https://gitlab.com/datadrivendiscovery/d3m/issues/102)
* Fixed standard return type of `log_likelihoods`, `log_likelihood`, `losses`, and `loss`
  primitive methods to support multi-target primitives.
* Clarified that `can_accept` receives primitive arguments and not just method arguments.
* Added `https://metadata.datadrivendiscovery.org/types/FilesCollection` for resources which are
  file collections. Also moved the main semantic type of file collection's values to the column.
* Fixed conversion of a simple list to a DataFrame.
* Added `https://metadata.datadrivendiscovery.org/types/Confidence` semantic type for columns
  representing confidence and `confidence_for` metadata which can help confidence column refer
  to the target column for which it is confidence for.
* Fixed default `can_accept` implementation to return type unwrapped from `CallResult`.
* Fixed `DataMetadata.remove` to preserve `for_value` value (and allow it to be set through the call).
* Fixed a case where automatically generated metadata overrode explicitly set existing metadata.
  [!25](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/25)
* Fixed `query_with_exceptions` metadata method to correctly return exceptions for
  deeper selectors.
* Added to `primitive_family`:
  * `SCHEMA_DISCOVERY`
  * `DATA_AUGMENTATION`
* Added to `algorithm_types`:
  * `HEURISTIC`
  * `MARKOV_RANDOM_FIELD`
  * `LEARNING_USING_PRIVILEGED_INFORMATION`
  * `APPROXIMATE_DATA_AUGMENTATION`
* Added `PrimitiveNotFittedError`, `DimensionalityMismatchError`, and `MissingValueError`
  exceptions to `d3m.exceptions`.
  [!22](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/22)
* Fixed setting semantic types for boundary columns.
  [#126](https://gitlab.com/datadrivendiscovery/d3m/issues/126) [!23](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/23)
* Added `video/avi` media type to lists of known media types.
* Fixed a type check which prevented an additional primitive argument to be of `Union` type.
* Fixed erroneous removal of empty dicts (`{}`) from metadata when empty dicts were
  explicitly stored in metadata.
  [#118](https://gitlab.com/datadrivendiscovery/d3m/issues/118)
* Made sure that conflicting entry points are resolved in a deterministic way.
* Made sure primitive metadata's `python_path` matches the path under which
  a primitive is registered under `d3m.primitives`. This also prevents
  a primitive to be registered twice at different paths in the namespace.
  [#4](https://gitlab.com/datadrivendiscovery/d3m/issues/4)
* Fixed a bug which prevented registration of primitives at deeper levels
  (e.g., `d3m.primitives.<name1>.<name2>.<primitive>`).
  [#121](https://gitlab.com/datadrivendiscovery/d3m/issues/121)

## v2018.6.5

* `Metadata` class got additional methods to manipulate metadata:
    * `remove(selector)` removes metadata at `selector`.
    * `query_with_exceptions(selector)` to return metadata for selectors which
      have metadata which differs from that of `ALL_ELEMENTS`.
    * `add_semantic_type`, `has_semantic_type`, `remove_semantic_type`,
      `get_elements_with_semantic_type` to help with semantic types.
    * `query_column`, `update_column`, `remove_column`, `get_columns_with_semantic_type`
      to make it easier to work with tabular data.

    [#55](https://gitlab.com/datadrivendiscovery/d3m/issues/55)
    [#78](https://gitlab.com/datadrivendiscovery/d3m/issues/78)

* Container `List` now inherits from a regular Python `list` and not from `typing.List`.
  It does not have anymore a type variable. Typing information is stored in `metadata`
  anyway (`structural_type`). This simplifies type checking (and improves performance)
  and fixes pickling issues.
  **Backwards incompatible.**
* `Hyperparams` class' `defaults` method now accepts optional `path` argument which
  allows one to fetch defaults from nested hyper-parameters.
* `Hyperparameters` class and its subclasses now have `get_default` method instead
  of a property `default`.
  **Backwards incompatible.**
* `Hyperparams` class got a new method `replace` which makes it easier to modify
  hyper-parameter values.
* `Set` hyper-parameter can now accept also a hyper-parameters configuration as elements
  which allows one to define a set of multiple hyper-parameters per each set element.
  [#94](https://gitlab.com/datadrivendiscovery/d3m/issues/94)
* Pipeline's `check` method now checks structural types of inputs and outputs and assures
  they match.
  [!19](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/19)
* `Set` hyper-parameter now uses tuple of unique elements instead of set to represent the set.
  This assures that the order of elements is preserved to help with reproducibility when
  iterating over a set.
  **Backwards incompatible.**
  [#109](https://gitlab.com/datadrivendiscovery/d3m/issues/109)
* `Set` hyper-parameter can now be defined without `max_samples` argument to allow a set
  without an upper limit on the number of elements.
  `min_samples` and `max_samples` arguments to `Set` constructor have been switched as
  a consequence, to have a more intuitive order.
  Similar changes have been done to `sample_multiple` method of hyper-parameters.
  **Backwards incompatible.**
  [#110](https://gitlab.com/datadrivendiscovery/d3m/issues/110)
* Core dependencies have been upgraded: `numpy==1.14.3`. `pytypes` is now a released version.
* When converting a numpy array with more than 2 dimensions to a DataFrame, higher dimensions are
  automatically converted to nested numpy arrays inside a DataFrame.
  [#80](https://gitlab.com/datadrivendiscovery/d3m/issues/80)
* Metadata is now automatically preserved when converting between container types.
  [#76](https://gitlab.com/datadrivendiscovery/d3m/issues/76)
* Basic metadata for data values is now automatically generated when using D3M container types.
  Value is traversed over its structure and `structural_type` and `dimension` with its `length`
  keys are populated. Some `semantic_types` are added in simple cases, and `dimension`'s
  `name` as well. In some cases analysis of all data to generate metadata can take time,
  so you might consider disabling automatic generation by setting `generate_metadata`
  to `False` in container's constructor or `set_for_value` calls and then manually populating
  necessary metadata.
  [#35](https://gitlab.com/datadrivendiscovery/d3m/issues/35)
  [!6](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/6)
  [!11](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/11)
* When reading D3M datasets, `media_types` metadata now includes proper media types
  for the column, and also media type for each particular row (file).
* D3M dataset and problem description parsing has been updated to 3.1.2 version:
    * `Dataset` class now supports loading `edgeList` resources.
    * `primitive_family` now includes `OBJECT_DETECTION`.
    * `task_type` now includes `OBJECT_DETECTION`.
    * `performance_metrics` now includes `PRECISION`, `RECALL`, `OBJECT_DETECTION_AVERAGE_PRECISION`.
    * `targets` of a problem description now includes `clusters_number`.
    * New metadata `boundary_for` can now describe for which other column
      a column is a boundary for.
    * Support for `realVector`, `json` and `geojson` column types.
    * Support for `boundingBox` column role.
    * New semantic types:
      * `https://metadata.datadrivendiscovery.org/types/EdgeList`
      * `https://metadata.datadrivendiscovery.org/types/FloatVector`
      * `https://metadata.datadrivendiscovery.org/types/JSON`
      * `https://metadata.datadrivendiscovery.org/types/GeoJSON`
      * `https://metadata.datadrivendiscovery.org/types/Interval`
      * `https://metadata.datadrivendiscovery.org/types/IntervalStart`
      * `https://metadata.datadrivendiscovery.org/types/IntervalEnd`
      * `https://metadata.datadrivendiscovery.org/types/BoundingBox`
      * `https://metadata.datadrivendiscovery.org/types/BoundingBoxXMin`
      * `https://metadata.datadrivendiscovery.org/types/BoundingBoxYMin`
      * `https://metadata.datadrivendiscovery.org/types/BoundingBoxXMax`
      * `https://metadata.datadrivendiscovery.org/types/BoundingBoxYMax`

    [#99](https://gitlab.com/datadrivendiscovery/d3m/issues/99)
    [#107](https://gitlab.com/datadrivendiscovery/d3m/issues/107)

* Unified the naming of attributes/features metafeatures to attributes.
  **Backwards incompatible.**
  [!13](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/13)
* Unified the naming of categorical/nominal metafeatures to categorical.
  **Backwards incompatible.**
  [!12](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/12)
* Added more metafeatures:
    * `pca`
    * `random_tree`
    * `decision_stump`
    * `naive_bayes`
    * `linear_discriminant_analysis`
    * `knn_1_neighbor`
    * `c45_decision_tree`
    * `rep_tree`
    * `categorical_joint_entropy`
    * `numeric_joint_entropy`
    * `number_distinct_values_of_numeric_features`
    * `class_probabilities`
    * `number_of_features`
    * `number_of_instances`
    * `canonical_correlation`
    * `entropy_of_categorical_features`
    * `entropy_of_numeric_features`
    * `equivalent_number_of_categorical_features`
    * `equivalent_number_of_numeric_features`
    * `mutual_information_of_categorical_features`
    * `mutual_information_of_numeric_features`
    * `categorical_noise_to_signal_ratio`
    * `numeric_noise_to_signal_ratio`

    [!10](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/10)
    [!14](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/14)
    [!17](https://gitlab.com/datadrivendiscovery/d3m/merge_requests/17)

* Added metafeatures for present values:
    * `number_of_instances_with_present_values`
    * `ratio_of_instances_with_present_values`
    * `number_of_present_values`
    * `ratio_of_present_values`

    [#84](https://gitlab.com/datadrivendiscovery/d3m/issues/84)

* Implemented interface for saving datasets.
  [#31](https://gitlab.com/datadrivendiscovery/d3m/issues/31)
* To remove a key in metadata, instead of using `None` value one should now use
  special `NO_VALUE` value.
  **Backwards incompatible.**
* `None` is now serialized to JSON as `null` instead of string `"None"`.
  **Could be backwards incompatible.**
* Unified naming and behavior of methods dealing with JSON and JSON-related
  data. Now across the package:
    * `to_json_structure` returns a structure with values fully compatible with JSON and serializable with default JSON serializer
    * `to_simple_structure` returns a structure similar to JSON, but with values left as Python values
    * `to_json` returns serialized value as JSON string

    **Backwards incompatible.**

* Hyper-parameters are now required to specify at least one
  semantic type from: `https://metadata.datadrivendiscovery.org/types/TuningParameter`,
  `https://metadata.datadrivendiscovery.org/types/ControlParameter`,
  `https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter`,
  `https://metadata.datadrivendiscovery.org/types/MetafeatureParameter`.
  **Backwards incompatible.**
* Made type strings in primitive annotations deterministic.
  [#93](https://gitlab.com/datadrivendiscovery/d3m/issues/93)
* Reimplemented primitives loading code to load primitives lazily.
  [#74](https://gitlab.com/datadrivendiscovery/d3m/issues/74)
* `d3m.index` module now has new and modified functions:
    * `search` now returns a list of Python paths of all potential
      primitives defined through entry points (but does not load them
      or checks if entry points are valid)
    * `get_primitive` loads and returns a primitive given its Python path
    * `get_primitive_by_id` returns a primitive given its ID, but a primitive
      has to be loaded beforehand
    * `get_loaded_primitives` returns a list of all currently loaded primitives
    * `load_all` tries to load all primitives
    * `register_primitive` now accepts full Python path instead of just suffix

    **Backwards incompatible.**
    [#74](https://gitlab.com/datadrivendiscovery/d3m/issues/74)

* Defined `model_features` primitive metadata to describe features supported
  by an underlying model. This is useful to allow easy matching between
  problem's subtypes and relevant primitives.
  [#88](https://gitlab.com/datadrivendiscovery/d3m/issues/88)
* Made hyper-parameter space of an existing `Hyperparams` subclass immutable.
  [#91](https://gitlab.com/datadrivendiscovery/d3m/issues/91)
* `d3m.index describe` command now accept `-s`/`--sort-keys` argument which
  makes all keys in the JSON output sorted, making output JSON easier to
  diff and compare.
* `can_accept` now gets a `hyperparams` object with hyper-parameters under
  which to check a method call. This allows `can_accept` to return a result
  based on control hyper-parameters.
  **Backwards incompatible.**
  [#81](https://gitlab.com/datadrivendiscovery/d3m/issues/81)
* Documented that all docstrings should be made according to
  [numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
  [#85](https://gitlab.com/datadrivendiscovery/d3m/issues/85)
* Added to semantic types:
  * `https://metadata.datadrivendiscovery.org/types/MissingData`
  * `https://metadata.datadrivendiscovery.org/types/InvalidData`
  * `https://metadata.datadrivendiscovery.org/types/RedactedTarget`
  * `https://metadata.datadrivendiscovery.org/types/RedactedPrivilegedData`
* Added to `primitive_family`:
  * `TIME_SERIES_EMBEDDING`
* Added to `algorithm_types`:
  * `IVECTOR_EXTRACTION`
* Removed `SparseDataFrame` from standard container types because it is being
  deprecated in Pandas.
  **Backwards incompatible.**
  [#95](https://gitlab.com/datadrivendiscovery/d3m/issues/95)
* Defined `other_names` metadata field for any other names a value might have.
* Optimized primitives loading time.
  [#87](https://gitlab.com/datadrivendiscovery/d3m/issues/87)
* Made less pickling of values when hyper-parameter has `Union` structural type.
  [#83](https://gitlab.com/datadrivendiscovery/d3m/issues/83)
* `DataMetadata.set_for_value` now first checks new value against the metadata, by default.
  **Could be backwards incompatible.**
* Added `NO_NESTED_VALUES` primitive precondition and effect.
  This allows primitive to specify if it cannot handle values where a container value
  contains nested other values with dimensions.

## v2018.4.18

* Added `pipeline.json` JSON schema to this package. Made `problem.json` JSON schema
  describing parsed problem description's schema. There is also a `d3m.metadata.pipeline`
  parser for pipelines in this schema and Python object to represent a pipeline.
  [#53](https://gitlab.com/datadrivendiscovery/d3m/issues/53)
* Updated README to make it explicit that for tabular data the first dimension
  is always rows and the second always columns, even in the case of a DataFrame
  container type.
  [#54](https://gitlab.com/datadrivendiscovery/d3m/issues/54)
* Made `Dataset` container type return Pandas `DataFrame` instead of numpy `ndarray`
  and in generaly suggest to use Pandas `DataFrame` as a default container type.
  **Backwards incompatible.**
  [#49](https://gitlab.com/datadrivendiscovery/d3m/issues/49)
* Added `UniformBool` hyper-parameter class.
* Renamed `FeaturizationPrimitiveBase` to `FeaturizationLearnerPrimitiveBase`.
  **Backwards incompatible.**
* Defined `ClusteringTransformerPrimitiveBase` and renamed `ClusteringPrimitiveBase`
  to `ClusteringLearnerPrimitiveBase`.
  **Backwards incompatible.**
  [#20](https://gitlab.com/datadrivendiscovery/d3m/issues/20)
* Added `inputs_across_samples` decorator to mark which method arguments
  are inputs which compute across samples.
  [#19](https://gitlab.com/datadrivendiscovery/d3m/issues/19)
* Converted `SingletonOutputMixin` to a `singleton` decorator. This allows
  each produce method separately to be marked as a singleton produce method.
  **Backwards incompatible.**
  [#17](https://gitlab.com/datadrivendiscovery/d3m/issues/17)
* `can_accept` can also raise an exception with information why it cannot accept.
  [#13](https://gitlab.com/datadrivendiscovery/d3m/issues/13)
* Added `Primitive` hyper-parameter to describe a primitive or primitives.
  Additionally, documented in docstrings better how to define hyper-parameters which
  use primitives for their values and how should such primitives-as-values be passed
  to primitives as their hyper-parameters.
  [#51](https://gitlab.com/datadrivendiscovery/d3m/issues/51)
* Hyper-parameter values can now be converted to and from JSON-compatible structure
  using `values_to_json` and `values_from_json` methods. Non-primitive values
  are pickled and stored as base64 strings.
  [#67](https://gitlab.com/datadrivendiscovery/d3m/issues/67)
* Added `Choice` hyper-parameter which allows one to define
  combination of hyper-parameters which should exists together.
  [#28](https://gitlab.com/datadrivendiscovery/d3m/issues/28)
* Added `Set` hyper-parameter which samples multiple times another hyper-parameter.
  [#52](https://gitlab.com/datadrivendiscovery/d3m/issues/52)
* Added `https://metadata.datadrivendiscovery.org/types/MetafeatureParameter`
  semantic type for hyper-parameters which control which meta-features are
  computed by the primitive.
  [#41](https://gitlab.com/datadrivendiscovery/d3m/issues/41)
* Added `supported_media_types` primitive metadata to describe
  which media types a primitive knows how to manipulate.
  [#68](https://gitlab.com/datadrivendiscovery/d3m/issues/68)
* Renamed metadata property `mime_types` to `media_types`.
  **Backwards incompatible.**
* Made pyarrow dependency a package extra. You can depend on it using
  `d3m[arrow]`.
  [#66](https://gitlab.com/datadrivendiscovery/d3m/issues/66)
* Added `multi_produce` method to primitive interface which allows primitives
  to optimize calls to multiple produce methods they might have.
  [#21](https://gitlab.com/datadrivendiscovery/d3m/issues/21)
* Added `d3m.utils.redirect_to_logging` context manager which can help
  redirect primitive's output to stdout and stderr to primitive's logger.
  [#65](https://gitlab.com/datadrivendiscovery/d3m/issues/65)
* Primitives can now have a dependency on static files and directories.
  One can use `FILE` and `TGZ` entries in primitive's `installation`
  metadata to ask the caller to provide paths those files and/or
  extracted directories through new `volumes` constructor argument.
  [#18](https://gitlab.com/datadrivendiscovery/d3m/issues/18)
* Core dependencies have been upgraded: `numpy==1.14.2`, `networkx==2.1`.
* LUPI quality in D3M datasets is now parsed into
  `https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData`
  semantic type for a column.
  [#61](https://gitlab.com/datadrivendiscovery/d3m/issues/61)
* Support for primitives using Docker containers has been put on hold.
  We are keeping a way to pass information about running containers to a
  primitive and defining dependent Docker images in metadata, but currently
  it is not expected that any runtime running primitives will run
  Docker containers for a primitive.
  [#18](https://gitlab.com/datadrivendiscovery/d3m/issues/18)
* Primitives do not have to define all constructor arguments anymore.
  This allows them to ignore arguments they do not use, e.g.,
  `docker_containers`.
  On the other side, when creating an instance of a primitive, one
  has now to check which arguments the constructor accepts, which is
  available in primitive's metadata:
  `primitive.metadata.query()['primitive_code'].get('instance_methods', {})['__init__']['arguments']`.
  [#63](https://gitlab.com/datadrivendiscovery/d3m/issues/63)
* Information about running primitive's Docker container has changed
  from just its address to a `DockerContainer` tuple containing both
  the address and a map of all exposed ports.
  At the same time, support for Docker has been put on hold so you
  do not really have to upgrade for this change anything and can simply
  remove the `docker_containers` argument from primitive's constructor.
  **Backwards incompatible.**
  [#14](https://gitlab.com/datadrivendiscovery/d3m/issues/14)
* Multiple exception classes have been defined in `d3m.exceptions`
  module and are now in use. This allows easier and more precise
  handling of exceptions.
  [#12](https://gitlab.com/datadrivendiscovery/d3m/issues/12)
* Fixed inheritance of `Hyperparams` class.
  [#44](https://gitlab.com/datadrivendiscovery/d3m/issues/44)
* Each primitive's class now automatically gets an instance of
  [Python's logging](https://docs.python.org/3/library/logging.html)
  logger stored into its ``logger`` class attribute. The instance is made
  under the name of primitive's ``python_path`` metadata value. Primitives
  can use this logger to log information at various levels (debug, warning,
  error) and even associate extra data with log record using the ``extra``
  argument to the logger calls.
  [#10](https://gitlab.com/datadrivendiscovery/d3m/issues/10)
* Made sure container data types can be serialized with Arrow/Plasma
  while retaining their metadata.
  [#29](https://gitlab.com/datadrivendiscovery/d3m/issues/29)
* `Scores` in `GradientCompositionalityMixin` replaced with `Gradients`.
  `Scores` only makes sense in a probabilistic context.
* Renamed `TIMESERIES_CLASSIFICATION`, `TIMESERIES_FORECASTING`, and
  `TIMESERIES_SEGMENTATION` primitives families to
  `TIME_SERIES_CLASSIFICATION`, `TIME_SERIES_FORECASTING`, and
  `TIME_SERIES_SEGMENTATION`, respectively, to match naming
  pattern used elsewhere.
  Similarly, renamed `UNIFORM_TIMESERIES_SEGMENTATION` algorithm type
  to `UNIFORM_TIME_SERIES_SEGMENTATION`.
  Compound words using hyphens are separated, but hyphens for prefixes
  are not separated. So "Time-series" and "Root-mean-squared error"
  become `TIME_SERIES` and `ROOT_MEAN_SQUARED_ERROR`
  but "Non-overlapping" and "Multi-class" are `NONOVERLAPPING` and `MULTICLASS`.
  **Backwards incompatible.**
* Updated performance metrics to include `PRECISION_AT_TOP_K` metric.
* Added to problem description parsing support for additional metric
  parameters and updated performance metric functions to use them.
  [#42](https://gitlab.com/datadrivendiscovery/d3m/issues/42)
* Merged `d3m_metadata`, `primitive_interfaces` and `d3m` repositories
  into `d3m` repository. This requires the following changes of
  imports in existing code:
    * `d3m_metadata` to `d3m.metadata`
    * `primitive_interfaces` to `d3m.primitive_interfaces`
    * `d3m_metadata.container` to `d3m.container`
    * `d3m_metadata.metadata` to `d3m.metadata.base`
    * `d3m_metadata.metadata.utils` to `d3m.utils`
    * `d3m_metadata.metadata.types` to `d3m.types`

    **Backwards incompatible.**
    [#11](https://gitlab.com/datadrivendiscovery/d3m/issues/11)

* Fixed computation of sampled values for `LogUniform` hyper-parameter class.
  [#47](https://gitlab.com/datadrivendiscovery/d3m/issues/47)
* When copying or slicing container values, metadata is now copied over
  instead of cleared. This makes it easier to propagate metadata.
  This also means one should make sure to update the metadata in the
  new container value to reflect changes to the value itself.
  **Could be backwards incompatible.**
* `DataMetadata` now has `set_for_value` method to make a copy of
  metadata and set new `for_value` value. You can use this when you
  made a new value and you want to copy over metadata, but you also
  want this value to be associated with metadata. This is done by
  default for container values.
* Metadata now includes SHA256 digest for primitives and datasets.
  It is computed automatically during loading. This should allow one to
  track exact version of primitive and datasets used.
  `d3m.container.dataset.get_d3m_dataset_digest` is a reference
  implementation of computing digest for D3M datasets.
  You can set `compute_digest` to `False` to disable this.
  You can set `strict_digest` to `True` to raise an exception instead
  of a warning if computed digest does not match one in metadata.
* Datasets can be now loaded in "lazy" mode: only metadata is loaded
  when creating a `Dataset` object. You can use `is_lazy` method to
  check if dataset iz lazy and data has not yet been loaded. You can use
  `load_lazy` to load data for a lazy object, making it non-lazy.
* There is now an utility metaclass `d3m.metadata.utils.AbstractMetaclass`
  which makes classes which use it automatically inherit docstrings
  for methods from the parent. Primitive base class and some other D3M
  classes are now using it.
* `d3m.metadata.base.CONTAINER_SCHEMA_VERSION` and
  `d3m.metadata.base.DATA_SCHEMA_VERSION` were fixed to point to the
  correct URI.
* Many `data_metafeatures` properties in metadata schema had type
  `numeric` which does not exist in JSON schema. They were fixed to
  `number`.
* Added to a list of known semantic types:
  `https://metadata.datadrivendiscovery.org/types/Target`,
  `https://metadata.datadrivendiscovery.org/types/PredictedTarget`,
  `https://metadata.datadrivendiscovery.org/types/TrueTarget`,
  `https://metadata.datadrivendiscovery.org/types/Score`,
  `https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint`,
  `https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData`,
  `https://metadata.datadrivendiscovery.org/types/PrivilegedData`.
* Added to `algorithm_types`: `ARRAY_CONCATENATION`, `ARRAY_SLICING`,
  `ROBUST_PRINCIPAL_COMPONENT_ANALYSIS`, `SUBSPACE_CLUSTERING`,
  `SPECTRAL_CLUSTERING`, `RELATIONAL_ALGEBRA`, `MULTICLASS_CLASSIFICATION`,
  `MULTILABEL_CLASSIFICATION`, `OVERLAPPING_CLUSTERING`, `SOFT_CLUSTERING`,
  `STRICT_PARTITIONING_CLUSTERING`, `STRICT_PARTITIONING_CLUSTERING_WITH_OUTLIERS`,
  `UNIVARIATE_REGRESSION`, `NONOVERLAPPING_COMMUNITY_DETECTION`,
  `OVERLAPPING_COMMUNITY_DETECTION`.

## v2018.1.26

* Test primitives updated to have `location_uris` metadata.
* Test primitives updated to have `#egg=` package URI suffix in metadata.
* Primitives (instances of their classes) can now be directly pickled
  and unpickled. Internally it uses `get_params` and `set_params` in
  default implementation. If you need to preserve additional state consider
  extending `__getstate__` and `__setstate__` methods.
* Added `RandomPrimitive` test primitive.
* Bumped `numpy` dependency to `1.14` and `pandas` to `0.22`.
* Added `https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter` as a known URI
  for `semantic_types` to help convey which hyper-parameters control the use of resources by the
  primitive.
  [#41](https://gitlab.com/datadrivendiscovery/metadata/issues/41)
* Fixed use of `numpy` values in `Params` and `Hyperparams`.
  [#39](https://gitlab.com/datadrivendiscovery/metadata/issues/39)
* Added `upper_inclusive` argument to `UniformInt`, `Uniform`, and `LogUniform` classes
  to signal that the upper bound is inclusive (default is exclusive).
  [#38](https://gitlab.com/datadrivendiscovery/metadata/issues/38)
* Made `semantic_types` and `description` keyword-only arguments in hyper-parameter description classes.
* Made all enumeration metadata classes have their instances be equal to their string names.
* Made sure `Hyperparams` subclasses can be pickled and unpickled.
* Improved error messages during metadata validation.
* Documented common metadata for primitives and data in the README.
* Added standard deviation to aggregate metadata values possible.
* Added `NO_JAGGED_VALUES` to `preconditions` and `effects`.
* Added to `algorithm_types`: `AGGREGATE_FUNCTION`, `AUDIO_STREAM_MANIPULATION`, `BACKWARD_DIFFERENCE_CODING`,
  `BAYESIAN_LINEAR_REGRESSION`, `CATEGORY_ENCODER`, `CROSS_VALIDATION`, `DISCRETIZATION`, `ENCODE_BINARY`,
  `ENCODE_ORDINAL`, `FEATURE_SCALING`, `FORWARD_DIFFERENCE_CODING`, `FREQUENCY_TRANSFORM`, `GAUSSIAN_PROCESS`,
  `HASHING`, `HELMERT_CODING`, `HOLDOUT`, `K_FOLD`, `LEAVE_ONE_OUT`, `MERSENNE_TWISTER`, `ORTHOGONAL_POLYNOMIAL_CODING`,
  `PASSIVE_AGGRESSIVE`, `PROBABILISTIC_DATA_CLEANING`, `QUADRATIC_DISCRIMINANT_ANALYSIS`, `RECEIVER_OPERATING_CHARACTERISTIC`,
  `RELATIONAL_DATA_MINING`, `REVERSE_HELMERT_CODING`, `SEMIDEFINITE_EMBEDDING`, `SIGNAL_ENERGY`, `SOFTMAX_FUNCTION`,
  `SPRUCE`, `STOCHASTIC_GRADIENT_DESCENT`, `SUM_CODING`, `TRUNCATED_NORMAL_DISTRIBUTION`, `UNIFORM_DISTRIBUTION`.
* Added to `primitive_family`: `DATA_GENERATION`, `DATA_VALIDATION`, `DATA_WRANGLING`, `VIDEO_PROCESSING`.
* Added `NoneType` to the list of data types allowed inside container types.
* For `PIP` dependencies specified by a `package_uri` git URI, an `#egg=package_name` URI suffix is
  now required.

## v2018.1.5

* Made use of the PyPI package official. Documented a requirement for
  `--process-dependency-links` argument during installation.
  [#27](https://gitlab.com/datadrivendiscovery/metadata/issues/27)
* Arguments `learning_rate` and `weight_decay` in `GradientCompositionalityMixin` renamed to
  `fine_tune_learning_rate` and `fine_tune_weight_decay`, respectively.
  `learning_rate` is a common hyper-parameter name.
  [#41](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/41)
* Added `https://metadata.datadrivendiscovery.org/types/TuningParameter` and
  `https://metadata.datadrivendiscovery.org/types/ControlParameter` as two known URIs for
  `semantic_types` to help convey which hyper-parameters are true tuning parameters (should be
  tuned during hyper-parameter optimization phase) and which are control parameters (should be
  determined during pipeline construction phase and are part of the logic of the pipeline).
* Made `installation` metadata optional. This allows local-only primitives.
  You can still register them into D3M namespace using `d3m.index.register_primitive`.
* Fixed serialization to JSON of hyper-parameters with `q` argument.
* Clarified that primitive's `PIP` dependency `package` has to be installed with `--process-dependency-link` argument
  enabled, and `package_uri` with both `--process-dependency-link` and `--editable`, so that primitives can have access
  to their git history to generate metadata.
* Only `git+http` and `git+https` URI schemes are allowed for git repository URIs for `package_uri`.
* Added to `algorithm_types`: `AUDIO_MIXING`, `CANONICAL_CORRELATION_ANALYSIS`, `DATA_PROFILING`, `DEEP_FEATURE_SYNTHESIS`,
  `INFORMATION_ENTROPY`, `MFCC_FEATURE_EXTRACTION`, `MULTINOMIAL_NAIVE_BAYES`, `MUTUAL_INFORMATION`, `PARAMETRIC_TRAJECTORY_MODELING`,
  `SIGNAL_DITHERING`, `SIGNAL_TO_NOISE_RATIO`, `STATISTICAL_MOMENT_ANALYSIS`, `UNIFORM_TIMESERIES_SEGMENTATION`.
* Added to `primitive_family`: `SIMILARITY_MODELING`, `TIMESERIES_CLASSIFICATION`, `TIMESERIES_SEGMENTATION`.

## v2017.12.27

* Documented `produce` method for `ClusteringPrimitiveBase` and added
  `ClusteringDistanceMatrixMixin`.
  [#18](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/18)
* Added `can_accept` class method to primitive base class and implemented its
  default implementation.
  [#20](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/20)
* "Distance" primitives now accept an extra argument instead of a tuple.
* `Params` should now be a subclass of `d3m.metadata.params.Params`, which is a
  specialized dict instead of a named tuple.
* Removed `Graph` class. There is no need for it anymore because we can identify
  them by having input type a NetworkX graph and through metadata discovery.
* Added `timeout` and `iterations` arguments to more methods.
* Added `forward` and `backward` backprop methods to `GradientCompositionalityMixin`
  to allow end-to-end backpropagation across diverse primitives.
  [#26](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/26)
* Added `log_likelihoods` method to `ProbabilisticCompositionalityMixin`.
* Constructor now accepts `docker_containers` argument with addresses of running
  primitive's Docker containers.
  [#25](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/25)
* Removed `CallMetadata` and `get_call_metadata` and changed so that some methods
  directly return new but similar `CallResult`.
  [#27](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/27)
* Documented how extra arguments to standard and extra methods can be defined.
* Documented that all arguments with the same name in all methods should have the
  same type. Arguments are per primitive not per method.
  [#29](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/29)
* Specified how to define extra "produce" methods which have same semantics
  as `produce` but different output types.
  [#30](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/30)
* Added `SingletonOutputMixin` to signal that primitive's output contains
  only one element.
  [#15](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/15)
* Added `get_loss_primitive` to allow accessing to the loss primitive
  being used.
* Moved `set_training_data` back to the base class.
  This breaks Liskov substitution principle.
  [#19](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/19)
* Renamed `__metadata__` to `metadata` attribute.
  [#23](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/23)
* `set_random_seed` method has been removed and replaced with a
  `random_seed` argument to the constructor, which is also exposed as an attribute.
  [#16](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/16)
* Primitives have now `hyperparams` attribute which returns a
  hyper-parameters object passed to the constructor.
  [#14](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/14)
* `Params` and `Hyperparams` are now required to be pickable and copyable.
  [#3](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/3)
* Primitives are now parametrized by `Hyperparams` type variable as well.
  Constructor now receives hyper-parameters as an instance as one argument
  instead of multiple keyword arguments.
  [#13](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/13)
* `LossFunctionMixin`'s `get_loss_function` method now returns a value from
  problem schema `Metric` enumeration.
* `LossFunctionMixin` has now a `loss` and `losses` methods which allows one
  to ask a primitive to compute loss for a given set of inputs and outputs using
  internal loss function the primitive is using.
  [#17](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/17)
* Added `Params` class.
* Removed `Graph` class in favor of NetworkX `Graph` class.
* Added `Metadata` class with subclasses and documented the use of selectors.
* Added `Hyperparams` class.
* Added `Dataset` class.
* "Sequences" have generally been renamed to "containers". Related code is also now under
  `d3m.container` and not under `d3m.metadata.sequence` anymore.
* `__metadata__` attribute was renamed to `metadata`.
* Package renamed from `d3m_types` to `d3m_metadata`.
* Added schemas for metadata contexts.
* A problem schema parsing and Python enumerations added in
  `d3m.metadata.problem` module.
* A standard set of container and base types have been defined.
* `d3m.index` command tool rewritten to support three commands: `search`, `discover`,
  and `describe`. See details by running `python -m d3m.index -h`.
* Package now requires Python 3.6.
* Repository migrated to gitlab.com and made public.

## v2017.10.10

* Made `d3m.index` module with API to register primitives into a `d3m.primitives` module
  and searches over it.
* `d3m.index` is also a command-line tool to list available primitives and automatically
  generate JSON annotations for primitives.
* Created `d3m.primitives` module which automatically populates itself with primitives
  using Python entry points.
