Pipeline
========

Pipeline is described as a DAG consisting of interconnected steps, where
steps can be primitives, or (nested) other pipelines. Pipeline has
data-flow semantics, which means that steps are not necessary executed
in the order they are listed, but a step can be executed when all its
inputs are available. Some steps can even be executed in parallel. On
the other hand, each step can use only previously defined outputs from
steps coming before in the order they are listed. In JSON, the following
is a sketch of its representation:

.. code:: yaml

    {
      "id": <UUID of the pipeline>,
      "schema": <a URI representing a schema and version to which pipeline description conforms>,
      "source": {
        "name": <string representing name of the author, team>,
        "contact": <contact information of author of the pipeline>,
        "from": <if pipeline was derived from another pipeline, or pipelines, which>
        ... # Any extra metadata author might want to add into the pipeline, like version,
            # name, and config parameters of the system which produced this pipeline.
      },
      "created": <timestamp when created>,
      "name": <human friendly name of the pipeline, if it exists>,
      "description": <human friendly description of the pipeline, if it exists>,
      "users": [
        {
          "id": <UUID for the user, if user is associated with the creation of the pipeline>,
          "reason": <textual description of what user did to create the pipeline>,
          "rationale": <textual description by the user of what the user did>
        }
      ],
      "inputs": [
        {
          "name": <human friendly name of the inputs>
        }
      ],
      "outputs": [
        {
          "name": <human friendly name of the outputs>,
          "data": <data reference, probably of an output of a step>
        }
      ],
      "steps": [
        {
          "type": "PRIMITIVE",
          "primitive": {
            "id": <ID of the primitive used in this step>,
            "version": <version of the primitive used in this step>,
            "python_path": <Python path of this primitive>,
            "name": <human friendly name of this primitive>,
            "digest": <digest of this primitive>
          },
          # Constructor arguments should not be listed here, because they can be automatically created from other
          # information. All these arguments are listed as kind "PIPELINE" in primitive's metadata.
          "arguments": {
             # A standard inputs argument used for both set_training_data and default "produce" method.
            "inputs": {
              "type": "CONTAINER",
              "data": <data reference, probably of an output of a step or pipeline input>
            },
             # A standard inputs argument, used for "set_training_data".
            "outputs": {
              "type": "CONTAINER",
              "data": <data reference, probably of an output of a step or pipeline input>
            },
            # An extra argument which takes as inputs outputs from another primitive in this pipeline.
            "extra_data": {
              "type": "CONTAINER",
              "data": <data reference, probably of an output of a step or pipeline input>
            },
            # An extra argument which takes as input a singleton output from another step in this pipeline.
            "offset": {
              "type": "DATA",
              "data": <data reference, probably of an output of a step or pipeline input>
            }
          },
          "outputs": [
            {
              # Data is made available by this step from default "produce" method.
              "id": "produce"
            },
            {
              # Data is made available by this step from an extra "produce" method, too.
              "id": "produce_score"
            }
          ],
          # Some hyper-parameters are not really tunable and should be fixed as part of pipeline definition. This
          # can be done here. Hyper-parameters listed here cannot be tuned or overridden during a run. Author of
          # a pipeline decides which hyper-parameters are which, probably based on their semantic type.
          # This is a map hyper-parameter names and their values using a similar format as arguments, but
          # allowing also PRIMITIVE and VALUE types.
          "hyperparams": {
            "loss": {
              "type": "PRIMITIVE",
              "data": <0-based index from steps identifying a primitive to pass in>
            },
            "column_to_operate_on": {
              "type": "VALUE",
              # Value is converted to a JSON-compatible value by hyper-parameter class.
              # It also knows how to convert it back.
              "data": 5
            },
            # A special case where a hyper-parameter can also be a list of primitives,
            # which are then passed to the \"Set\" hyper-parameter class.
            "ensemble": {
              "type": "PRIMITIVE",
              "data": [
                <0-based index from steps identifying a primitive to pass in>,
                <0-based index from steps identifying a primitive to pass in>
              ]
            }
          },
          "users": [
            {
              "id": <UUID for the user, if user is associated with selection of this primitive/arguments/hyper-parameters>,
              "reason": <textual description of what user did to select this primitive>,
              "rationale": <textual description by the user of what the user did>
            }
          ]
        },
        {
          "type": "SUBPIPELINE",
          "pipeline": {
            "id": <UUID of a pipeline to run as this step>
          },
          # For example: [{"data": "steps.0.produce"}] would map the data reference "steps.0.produce" of
          # the outer pipeline to the first input of a sub-pipeline.
          "inputs": [
            {
              "data": <data reference, probably of an output of a step or pipeline input, mapped to sub-pipeline's inputs in order>
            }
          ],
          # For example: [{"id": "predictions"}] would map the first output of a sub-pipeline to a data
          # reference "steps.X.predictions" where "X" is the step number of a given sub-pipeline step.
          "outputs": [
            {
              "id": <ID to be used in data reference, mapping sub-pipeline's outputs in order>
            }
          ]
        },
        {
          # Used to represent a pipeline template which can be used to generate full pipelines. Not to be used in
          # the metalearning context. Additional properties to further specify the placeholder constraints are allowed.
          "type": "PLACEHOLDER",
          # A list of inputs which can be used as inputs to resulting sub-pipeline.
          # Resulting sub-pipeline does not have to use all the inputs, but it cannot use any other inputs.
          "inputs": [
            {
              "data": <data reference, probably of an output of a step or pipeline input>
            }
          ],
          # A list of outputs of the resulting sub-pipeline.
          # Their (allowed) number and meaning are defined elsewhere.
          "outputs": [
            {
              "id": <ID to be used in data reference, mapping resulting sub-pipeline's outputs in order>
            }
          ]
        }
      ]
    }

``id`` uniquely identifies this particular database document.

Pipeline describes how inputs are computed into outputs. In most cases
inputs are :class:`~d3m.container.dataset.Dataset` container values and
outputs are predictions as Pandas :class:`~d3m.container.pandas.DataFrame` container
values in `Lincoln Labs predictions
format <https://gitlab.com/datadrivendiscovery/data-supply/blob/shared/documentation/problemSchema.md#predictions-file>`__,
and, during training, potentially also internal losses/scores. The same
pipeline is used for both training and predicting.

Pipeline description contains many *data references*. Data reference is
just a string which identifies an output of a step or a pipeline input
and forms a data-flow connection between data available and an input to
a step. It is recommended to be a string of the following forms:

-  ``steps.<number>.<id>`` — ``number`` identifies the step in the list
   of steps (0-based) and ``id`` identifies the name of a produce method
   of the primitive, or the output of a pipeline step
-  ``inputs.<number>`` — ``number`` identifies the pipeline input
   (0-based)
-  ``outputs.<number>`` — ``number`` identifies the pipeline output
   (0-based)

Inputs in the context of metalearning are expected to be datasets, and
the order of inputs match the order of datasets in a pipeline run. (In
other contexts, like TA2-TA3 API, inputs might be something else, for
example a pipeline can consist of just one primitive a TA3 wants to run
on a particular input.)

Remember that each primitive has a set of arguments it takes as a whole,
combining all the arguments from all its methods. Each argument
(identified by its name) can have only one value associated with it and
any method accepting that argument receives that value. Once all values
for all arguments for a method are available, that method can be called.

Remember as well that each primitive can have multiple "produce"
methods. These methods can be called after a primitive has been fitted.
In this way a primitive can have multiple outputs, for each "produce"
method one.

Placeholders can be used to define pipeline templates to be used outside
of the metalearning context. A placeholder is replaced with a pipeline
step to form a pipeline. Restrictions of placeholders may apply on the
number of them, their position, allowed inputs and outputs, etc.

.. _pipeline-description-example:

Pipeline description example
----------------------------

The following example uses the core package and the `common primitives
repo <https://gitlab.com/datadrivendiscovery/common-primitives>`__, this
example provides the basic knowledge to build a pipeline in memory. This
specific example creates a pipeline for classification task.

.. code:: python

    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep

    # -> dataset_to_dataframe -> column_parser -> extract_columns_by_semantic_types(attributes) -> imputer -> random_forest
    #                                             extract_columns_by_semantic_types(targets)    ->            ^

    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 1: dataset_to_dataframe
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 2: column_parser
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 3: extract_columns_by_semantic_types(attributes)
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                              data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    pipeline_description.add_step(step_2)

    # Step 4: extract_columns_by_semantic_types(targets)
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                              data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    pipeline_description.add_step(step_3)

    attributes = 'steps.2.produce'
    targets = 'steps.3.produce'

    # Step 5: imputer
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # Step 6: random_forest
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.regression.random_forest.SKlearn'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.5.produce')

    # Output to YAML
    print(pipeline_description.to_yaml())

Pipeline Run
------------

:mod:`d3m.metadata.pipeline_run` module contains the classes that represent the Pipeline Run. The Pipeline Run was
introduced to ensure that pipeline execution could be captured and duplicated. To accomplish this, the problem doc,
hyperparameter settings and any other variables to the pipeline execution phases are captured by the Pipeline Run.

The Pipeline Run is generated during pipeline execution:

::

    $ python3 -m d3m runtime fit-produce -p pipeline.json -r problem/problemDoc.json -i dataset_TRAIN/datasetDoc.json \
         -t dataset_TEST/datasetDoc.json -o results.csv -O pipeline_run.yml

 In JSON, the following is a sketch of the Pipeline Run representation in two phases for the above fit-produce call:

.. code:: yaml

    context: <The run context for this phase (TESTING for example)>
    datasets:
      <ID and digest of the dataset>
    end: <timestamp of when this phase ended>
    environment:
      <Details about the machine the phase was performed on>
    id: e3187585-cf8b-5e31-9435-69907912c3ca <id of this pipeline run instance>
    pipeline:
      <ID and Digest of the pipeline>
    problem:
       <ID and digest of the problem>
    random_seed: <Random seed value, 0 if none provided>
    run:
      is_standard_pipeline: true
      phase: FIT
      results:
        <Results of the fit phase>
    schema: https://metadata.datadrivendiscovery.org/schemas/v0/pipeline_run.json
    start: <timestamp of when this phase started>
    status:
      state: <Whether this stage completed successfully or not>
    steps:
      <Details of each step (primitive) in this stage of running the pipeline, parameters start/end times plus success/failure>
    --- <This indicates a divider between phases like fit and produce in this example>
    context: <The run context for this phase (TESTING for example)>
    datasets:
      <ID and digest of the dataset>
    end: <timestamp of when this phase ended>
    environment:
      <Details about the machine the phase was performed on>
    id: b2e9b591-c332-5bc5-815e-d1ec73ecdb06 <id of this pipeline run instance>
    pipeline:
      <ID and Digest of the pipeline>
    previous_pipeline_run:
      id: e3187585-cf8b-5e31-9435-69907912c3ca <id of the previous pipeline run instance (see above)>
    problem:
      <ID and digest of the problem>
    random_seed: <Random seed value, 0 if none provided>
    run:
      is_standard_pipeline: true
      phase: PRODUCE
      results:
        <Results of the fit phase>
      scoring:
        datasets:
          <ID and Digest of the scoring dataset>
        end: <timestamp of when the scoring phase ended>
        pipeline:
          <ID and Digest of the pipeline>
        random_seed: <Random seed value, 0 if none provided>
        start: <timestamp of when the scoring phase started>
        status:
          state: <Whether this scoring stage completed successfully or not>
        steps:
          <Details of each step (primitive) in this stage of running the pipeline, parameters start/end times plus success/failure>
    schema: https://metadata.datadrivendiscovery.org/schemas/v0/pipeline_run.json
    start: <timestamp of when this phase started>
    status:
      state: <Whether this stage completed successfully or not>
    steps:
      <Details of each step (primitive) in this stage of running the pipeline, parameters start/end times plus success/failure>

The d3m module has a call that supports actions Pipeline Run:

::

    $ python3 -m d3m pipeline-run --help

Currently there is only one command available which validates a Pipeline Run:

::

    $ python3 -m d3m pipeline-run validate pipeline_run.yml

The Reference Runtime offers a way to pass an existing Pipeline Run file to a runtime command to allow it to be rerun.
Here is an example of this for the fit-produce call:

::

    $ python3 -m d3m runtime fit-produce -u pipeline_run.yml

Here is the guidance from the help menu:

::

      -u INPUT_RUN, --input-run INPUT_RUN
                        path to a pipeline run file with configuration, use
                        "-" for stdin


Reference runtime
-----------------

:mod:`d3m.runtime` module contains a reference runtime for pipelines. This
module also has an extensive command line interface you can access
through ``python3 -m d3m runtime``.

Example of fitting and producing a pipeline with Runtime:

.. code:: python

    from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
    from d3m.container.dataset import Dataset
    from d3m.runtime import Runtime

    # Loading problem description.
    problem_description = problem.parse_problem_description('problemDoc.json')

    # Loading dataset.
    path = 'file://{uri}'.format(uri=os.path.abspath('datasetDoc.json'))
    dataset = Dataset.load(dataset_uri=path)

    # Loading pipeline description file.
    with open('pipeline_description.json', 'r') as file:
        pipeline_description = pipeline_module.Pipeline.from_json(string_or_file=file)

    # Creating an instance on runtime with pipeline description and problem description.
    runtime = Runtime(pipeline=pipeline_description, problem_description=problem_description, context=metadata_base.Context.TESTING)

    # Fitting pipeline on input dataset.
    fit_results = runtime.fit(inputs=[dataset])
    fit_results.check_success()

    # Producing results using the fitted pipeline.
    produce_results = runtime.produce(inputs=[dataset])
    produce_results.check_success()

    print(produce_results.values)

Also, the Runtime provides a very useful set of tools to run pipelines
on the terminal, here is a basic example of how to fit and produce a
pipeline like the previous example:

::

    $ python3 -m d3m runtime fit-produce -p pipeline.json -r problem/problemDoc.json -i dataset_TRAIN/datasetDoc.json -t dataset_TEST/datasetDoc.json -o results.csv -O pipeline_run.yml

For more information about the usage:

::

    $ python3 -m d3m runtime --help
