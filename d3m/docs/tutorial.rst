Advanced Tutorial
=================

This tutorial assumes the reader is familiar with d3m ecosystem in general.
If not, please refer to other sections of `documentation`_ first, e.g.,
:ref:`quickstart`.

.. _documentation: https://docs.datadrivendiscovery.org

Overview of building a primitive
--------------------------------

1. :ref:`Recognize the base class of a primitive <primitive-class>`.

2. :ref:`Identify the input and output container types <input-output-types>`.

3. :ref:`Define metadata for each primitive <tutorial-primitive-metadata>`.

4. :ref:`Write a unit test to verify the primitive functions <unit-tests>`.

5. :ref:`Generate the primitive annotation for the primitive <primitive-annotation>`.

6. :ref:`Write pipeline for demonstrating primitive functionality <example-pipeline>`.

7. :ref:`Advanced: Primitive might use static files <static-files>`.

.. _primitive-class:

Primitive class
---------------

There are a variety of :py:mod:`primitive interfaces/classes <d3m.primitive_interfaces>` available. As an example,
a primitive doing just attribute extraction without requiring any fitting, a :py:class:`~d3m.primitive_interfaces.transformer.TransformerPrimitiveBase`
from :py:mod:`~d3m.primitive_interfaces.transformer` module can be used.

Each primitives can have it's own :py:mod:`hyper-parameters <d3m.metadata.hyperparams>`. Some example hyper-parameter types one can use to describe
primitive's hyper-parameters are: :py:class:`~d3m.metadata.hyperparams.Constant`, :py:class:`~d3m.metadata.hyperparams.UniformBool`,
:py:class:`~d3m.metadata.hyperparams.UniformInt`, :py:class:`~d3m.metadata.hyperparams.Choice`, :py:class:`~d3m.metadata.hyperparams.List`.

Also, each hyper-parameter should be defined as one or more of the four :ref:`hyper-parameter semantic types <hyperparameters>`:

* `https://metadata.datadrivendiscovery.org/types/TuningParameter <https://metadata.datadrivendiscovery.org/types/TuningParameter>`__
* `https://metadata.datadrivendiscovery.org/types/ControlParameter <https://metadata.datadrivendiscovery.org/types/ControlParameter>`__
* `https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter <https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter>`__
* `https://metadata.datadrivendiscovery.org/types/MetafeatureParameter <https://metadata.datadrivendiscovery.org/types/MetafeatureParameter>`__

Example
~~~~~~~

.. code:: python

    from d3m.primitive_interfaces import base, transformer
    from d3m.metadata import base as metadata_base, hyperparams

    __all__ = ('ExampleTransformPrimitive',)


    class Hyperparams(hyperparams.Hyperparams):
        learning_rate = hyperparams.Uniform(lower=0.0, upper=1.0, default=0.001, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ])
        clusters = hyperparams.UniformInt(lower=1, upper=100, default=10, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
        ])


    class ExampleTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
        The docstring is very important and must to be included. It should contain
        relevant information about the hyper-parameters, primitive functionality, etc.
        """

        def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
            pass

.. _input-output-types:

Input/Output types
------------------

The acceptable inputs/outputs of a primitive must be pre-defined. D3M supports a variety of
standard input/output :ref:`container types <container_types>` such as:

- ``pandas.DataFrame`` (as :py:class:`d3m.container.pandas.DataFrame`)

- ``numpy.ndarray`` (as :py:class:`d3m.container.numpy.ndarray`)

- ``list`` (as :py:class:`d3m.container.list.List`)

.. note::
    Even thought D3M container types behave mostly as standard types, the D3M container types must be used for inputs/outputs, because D3M container types support D3M metadata.

Example
~~~~~~~

.. code:: python

    from d3m import container

    Inputs  = container.DataFrame
    Outputs = container.DataFrame


    class ExampleTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        ...

.. note::
    When returning the output DataFrame, its metadata should be updated with the correct semantic and structural types.

Example
~~~~~~~

.. code:: python

    # Update metadata for each DataFrame column.
    for column_index in range(outputs.shape[1]):
        column_metadata = {}
        column_metadata['structural_type'] = type(1.0)
        column_metadata['name'] = "column {i}".format(i=column_index)
        column_metadata["semantic_types"] = ("http://schema.org/Float", "https://metadata.datadrivendiscovery.org/types/Attribute",)
        outputs.metadata = outputs.metadata.update((metadata_base.ALL_ELEMENTS, column_index), column_metadata)

.. _tutorial-primitive-metadata:

Primitive Metadata
------------------

It is very crucial to define :ref:`primitive metadata <primitive-metadata>` for the primitive properly.
Primitive metadata can be used by TA2 systems to metalearn about primitives and in general decide which primitive to use when.

Example
~~~~~~~

.. code:: python

    from d3m.primitive_interfaces import base, transformer
    from d3m.metadata import base as metadata_base, hyperparams

    __all__ = ('ExampleTransformPrimitive',)

    class ExampleTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
        Docstring.
        """

        metadata = metadata_base.PrimitiveMetadata({
            'id': <Unique-ID, generated using UUID>,
            'version': <Primitive-development-version>,
            'name': <Primitive-Name>,
            'python_path': 'd3m.primitives.<>.<>.<>' # Must match path in setup.py,
            'source': {
                'name': <Project-maintainer-name>,
                'uris': [<GitHub-link-to-project>],
                'contact': 'mailto:<Author E-Mail>'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+<git-link-to-project>@{git_commit}#egg=<Package_name>'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                # Check https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.algorithm_types for all available algorithm types.
                # If algorithm type s not available a Merge Request should be made to add it to core package.
                metadata_base.PrimitiveAlgorithmType.<Choose-the-algorithm-type-that-best-describes-the-primitive>,
            ],
            # Check https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.primitive_family for all available primitive family types.
            # If primitive family is not available a Merge Request should be made to add it to core package.
            'primitive_family': metadata_base.PrimitiveFamily.<Choose-the-primitive-family-that-closely-associates-to-the-primitive>
        })

        ...

.. _unit-tests:

Unit tests
----------

Once the primitives are constructed, unit testing must be done to see if the
primitive works as intended.

**Sample Setup**

.. code:: python

    import os
    import unittest

    from d3m.container import dataset
    from d3m.metadata import base as metadata_base
    from common_primitives import dataset_to_dataframe

    from example_primitive import ExampleTransformPrimitive


    class ExampleTransformTest(unittest.TestCase):
        def test_happy_path():
            # Load a dataset.
            # Datasets can be obtained from: https://datasets.datadrivendiscovery.org/d3m/datasets
            base_path = '../datasets/training_datasets/seed_datasets_archive/'
            dataset_doc_path = os.path.join(base_path, '38_sick_dataset', 'datasetDoc.json')
            dataset = dataset.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

            dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
            dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
            dataframe = dataframe_primitive.produce(inputs=dataset).value

            # Call example transformer.
            hyperparams_class = SampleTransform.metadata.get_hyperparams()
            primitive  = SampleTransform(hyperparams=hyperparams_class.defaults())
            test_out   = primitive.produce(inputs=dataframe).value

            # Write assertions to make sure that the output (type, shape, metadata) is what is expected.
            self.assertEqual(...)

            ...


    if __name__ == '__main__':
        unittest.main()

It is recommended to do the testing inside the D3M Docker container:

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
    cd /mnt/d3m/example_primitive
    python3 primitive_name_test.py

.. _primitive-annotation:

Primitive annotation
--------------------

Once primitive is constructed and unit testing is successful, the
final step in building a primitive is to generate the primitive annotation
which will be indexed and used by D3M.

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
    cd /mnt/d3m/example_primitive
    pip3 install -e .
    python3 -m d3m index describe -i 4 <primitive_name>

Alternatively, a `helper script <https://gitlab.com/datadrivendiscovery/docs-quickstart/-/blob/master/quickstart_primitives/generate-primitive-json.py>`__
can be used to generate primitive annotations as well.
This can be more convenient when having to manage multiple primitives.
In this case, generating the primitive annotation is done as follows:

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
    cd /mnt/d3m/example_primitive
    pip3 install -e .
    python3 generate-primitive-json.py ...

.. _example-pipeline:

Example pipeline
----------------

After building custom primitives, it has to be used in an example pipeline and run using one of
D3M seed datasets in order to be integrated with other indexed D3M primitives.

The essential elements of pipelines are:

``Dataset Denormalizer -> Dataset Parser -> Data Cleaner (If necessary) -> Feature Extraction -> Classifier/Regressor -> Output``

An example code of building pipeline is shown below:

.. code:: python

    # D3M dependencies
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep

    # Common Primitives
    from common_primitives.column_parser import ColumnParserPrimitive
    from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
    from common_primitives.extract_columns_semantic_types import ExtractColumnsBySemanticTypesPrimitive

    # Testing primitive
    from quickstart_primitives.sample_primitive1.input_to_output import InputToOutputPrimitive

    # Pipeline
    pipeline = Pipeline()
    pipeline.add_input(name='inputs')

    # Step 0: DatasetToDataFrame (Dataset Denormalizer)
    step_0 = PrimitiveStep(primitive_description=DatasetToDataFramePrimitive.metadata.query())
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Step 1: Custom primitive
    step_1 = PrimitiveStep(primitive=InputToOutputPrimitive)
    step_1.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # Step 2: Column Parser (Dataset Parser)
    step_2 = PrimitiveStep(primitive_description=ColumnParserPrimitive.metadata.query())
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: Extract Attributes (Feature Extraction)
    step_3 = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'] )
    pipeline.add_step(step_3)

    # Step 4: Extract Targets (Feature Extraction)
    step_4 = PrimitiveStep(primitive_description=ExtractColumnsBySemanticTypesPrimitive.metadata.query())
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_4.add_output('produce')
    step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'] )
    pipeline.add_step(step_4)

    attributes = 'steps.3.produce'
    targets    = 'steps.4.produce'

    # Step 6: Imputer (Data Cleaner)
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
    step_5.add_output('produce')
    pipeline.add_step(step_5)

    # Step 7: Classifier
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.decision_tree.SKlearn'))
    step_6.add_argument(name='inputs',  argument_type=ArgumentType.CONTAINER,  data_reference='steps.5.produce')
    step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
    step_6.add_output('produce')
    pipeline.add_step(step_6)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.6.produce')

    # print(pipeline.to_json())
    with open('./pipeline.json', 'w') as write_file:
        write_file.write(pipeline.to_json(indent=4, sort_keys=False, ensure_ascii=False))

Once pipeline is constructed and the pipeline's JSON file is generated, the pipeline is run using
``python3 -m d3m runtime`` command.
Successfully running the pipeline validates that the primitive is working as intended.

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
      /bin/bash -c "cd /mnt/d3m; \
        pip3 install -e .; \
        cd pipelines; \
        python3 -m d3m runtime fit-produce \
                --pipeline pipeline.json \
                --problem /datasets/seed_datasets_current/38_sick/TRAIN/problem_TRAIN/problemDoc.json \
                --input /datasets/seed_datasets_current/38_sick/TRAIN/dataset_TRAIN/datasetDoc.json \
                --test-input /datasets/seed_datasets_current/38_sick/TEST/dataset_TEST/datasetDoc.json \
                --output 38_sick_results.csv \
                --output-run pipeline_run.yml; \
        exit"

.. _static-files:

Advanced: Primitive with static files
-------------------------------------

When building primitives that uses external/static files i.e. pre-trained weights, the
metadata for the primitive must be properly define such dependency.
The static file can be hosted anywhere based on your preference, as long as the URL to the file is a direct download link. It must
be public so that users of your primitive can access the file. Be sure to keep the URL available, as
the older version of the primitive could potentially start failing if URL stops resolving.

.. note::
    Full code of this section can be found in the `quickstart repository <https://gitlab.com/datadrivendiscovery/docs-quickstart>`__.

Below is a description of primitive metadata definition required, named ``_weights_configs`` for
each static file.

.. code:: python

    _weights_configs = [{
        'type': 'FILE',
        'key': '<Weight File Name>',
        'file_uri': '<URL to directly Download the Weight File>',
        'file_digest':'sha256sum of the <Weight File>',
    }]


This ``_weights_configs`` should be directly added to the ``INSTALLATION`` field of the primitive metadata.

.. code:: python

    from d3m.primitive_interfaces import base, transformer
    from d3m.metadata import base as metadata_base, hyperparams

    __all__ = ('ExampleTransform',)

    class ExampleTransform(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
        """
        Docstring.
        """

        _weights_configs = [{
            'type': 'FILE',
            'key': '<Weight File Name>',
            'file_uri': '<URL to directly Download the Weight File>',
            'file_digest':'sha256sum of the <Weight File>',
        }]

        metadata = ...
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+<git-link-to-project>@{git_commit}#egg=<Package_name>'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }] + _weights_configs,
            ...

        ...

After the primitive metadata definition, it is important to include code to return the path of files.
An example is given as follows:

.. code:: python

    def _find_weights_path(self, key_filename):
        if key_filename in self.volumes:
            weight_file_path = self.volumes[key_filename]
        else:
            weight_file_path = os.path.join('.', self._weights_configs['file_digest'], key_filename)

        if not os.path.isfile(weight_file_path):
            raise ValueError(
                "Can't get weights file from volumes by key '{key_filename}' and at path '{path}'.".format(
                    key_filename=key_filename,
                    path=weight_file_path,
                ),
            )

        return weight_file_path

In this example code,  ``_find_weights_path`` method will try to find the static files from volumes based on weight file key.
If it cannot be found (e.g., runtime was not provided with static files), then it looks into the current directory.
The latter fallback is useful during development.

To run a pipeline with such primitive, you have to download static files and provide them to the runtime:

.. code:: shell

    docker run --rm -v /home/foo/d3m:/mnt/d3m -it \
      registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
      /bin/bash -c "cd /mnt/d3m; \
        pip3 install -e .; \
        cd pipelines; \
        mkdir /static
        python3 -m d3m index download -p d3m.primitives.path.of.Primitive -o /static; \
        python3 -m d3m runtime --volumes /static fit-produce \
                --pipeline feature_pipeline.json \
                --problem /datasets/seed_datasets_current/22_handgeometry/TRAIN/problem_TRAIN/problemDoc.json \
                --input /datasets/seed_datasets_current/22_handgeometry/TRAIN/dataset_TRAIN/datasetDoc.json \
                --test-input /datasets/seed_datasets_current/22_handgeometry/TEST/dataset_TEST/datasetDoc.json \
                --output 22_handgeometry_results.csv \
                --output-run feature_pipeline_run.yml; \
        exit"

The static files will be downloaded and stored locally based on ``file_digest`` of ``_weights_configs``.
In this way we don't duplicate same files used by multiple primitives:

.. code:: shell

    mkdir /static
    python3 -m d3m index download -p d3m.primitives.path.of.Primitive -o /static

``-p`` optional argument to download static files for a particular primitive, matching on its Python path.
``-o`` optional argument to download the static files into a common folder. If not provided, they are
downloaded into the current directory.

After the download, the file structure is given as follows::

    /static/
      <file_digest>/
        <file>
      <file_digest>/
        <file>
      ...
      ...
