.. _quickstart:

TA1 quick-start guide
=====================

This aims to be a tutorial, or a quick-start guide, for
newcomers to the D3M project who are interested in writing TA1 primitives.
It is not meant to be a comprehensive
guide to everything about D3M, or even just TA1. The goal here is for
the reader to be able to write a new, simple, but working primitive by
the end of this tutorial. To achieve this goal, this tutorial is divided
into several sections:

Important links
---------------

First, here is a list of some important links that should help you with
reference and instructional material beyond this quick start guide. Be
aware also that the d3m core package source code has extensive docstrings that
:ref:`you may find helpful <api-reference>`.

-  Documentation of the whole D3M program:
   `https://docs.datadrivendiscovery.org <https://docs.datadrivendiscovery.org>`__
-  Common primitives:
   `https://gitlab.com/datadrivendiscovery/common-primitives <https://gitlab.com/datadrivendiscovery/common-primitives>`__
-  Public datasets:
   `https://datasets.datadrivendiscovery.org/d3m/datasets <https://datasets.datadrivendiscovery.org/d3m/datasets>`__
-  Docker images:
   `https://docs.datadrivendiscovery.org/docker.html <https://docs.datadrivendiscovery.org/docker.html>`__
-  Index of TA1, TA2, TA3 repositories:
   `https://github.com/darpa-i2o/d3m-program-index <https://github.com/darpa-i2o/d3m-program-index>`__
-  :ref:`primitive-good-citizen`

.. _overview-of-primitives-and-pipelines:

Overview of primitives and pipelines
------------------------------------

Let's start with basic definitions in order for us to understand a
little bit better what happens when we run a pipeline later in the
tutorial.

A *pipeline* is basically a series of steps that are executed in order
to solve a particular *problem* (such as prediction based on historical
data). A step of a pipeline is usually a *primitive* (a step can be
something else, however, like a sub-pipeline, but for the purposes of
this tutorial, assume that each step is a primitive): something that
individually could, for example, transform data into another format, or
fit a model for prediction. There are many types of primitives (see the
`primitives index repo`_ for the full
list of available primitives). In a pipeline, the steps must be arranged
in a way such that each step must be able to read the data in the format
produced by the preceding step.

.. _primitives index repo: https://gitlab.com/datadrivendiscovery/primitives

For this tutorial, let's try to use the example pipeline that comes with
a primitive called
``d3m.primitives.classification.logistic_regression.SKlearn`` to predict
baseball hall-of-fame players, based on their stats (see the
`185_baseball dataset <https://datasets.datadrivendiscovery.org/d3m/datasets/-/tree/master/training_datasets/seed_datasets_archive/185_baseball>`__).

Let's take a look at the example pipeline. Many example pipelines can be found
in `primitives index repo`_ where they demonstrate how to use particular primitives.
At the time of this writing, an example pipeline can be found `here
<https://gitlab.com/datadrivendiscovery/primitives/blob/master/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipelines/862df0a2-2f87-450d-a6bd-24e9269a8ba6.json>`__,
but this repository's directory names and files periodically change, so it is
prudent to see how to navigate to this file too.

The index is organized as:
- ``v2020.1.9`` (version of the core package of the index, changes periodically)
- ``JPL`` (the organization that develops/maintains the primitive)
- ``d3m.primitives.classification.logistic_regression.SKlearn`` (the python path of the actual primitive)
- ``2019.11.13`` (the version of this primitive, changes periodically)
- ``pipelines``
- ``862df0a2-2f87-450d-a6bd-24e9269a8ba6.json`` (actual pipeline description filename, changes periodically)

Early on in this JSON document, you will see a list called ``steps``. This
is the actual list of primitive steps that run one after another in a
pipeline. Each step has the information about the primitive, as well as
arguments, outputs, and hyper-parameters, if any. This specific pipeline
has 5 steps (the ``d3m.primitives`` prefix is omitted in the following
list):

- ``data_transformation.dataset_to_dataframe.Common``
- ``data_transformation.column_parser.Common``
- ``data_cleaning.imputer.SKlearn``
- ``classification.logistic_regression.SKlearn``
- ``data_transformation.construct_predictions.Common``

Now let's take a look at the first primitive step in that pipeline. We
can find the source code of this primitive in the common-primitives repo
(`common_primitives/dataset_to_dataframe.py
<https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataset_to_dataframe.py>`__).
Take a look particularly at the ``produce`` method. This is essentially
what the primitive does. Try to do this for the other primitive steps in
the pipeline as well - take a cursory look at what each one essentially
does (note that for the actual classifier primitive, you should look at
the ``fit`` method as well to see how the model is trained). Primitives
whose python path suffix is ``*.Common`` is in the `common primitives <https://gitlab.com/datadrivendiscovery/common-primitives>`__
repository, and those that have a ``*.SKlearn`` suffix is in the
`sklearn-wrap <https://gitlab.com/datadrivendiscovery/sklearn-wrap>`__ repository (checkout the `dist <https://gitlab.com/datadrivendiscovery/sklearn-wrap/-/tree/dist>`__ branch,
to which primitives are being generated).

If you're having a hard time looking for the correct source file, you can try
taking the primitive ``id`` from the primitive step description in the
pipeline, and ``grep`` for it. For example, if you were
looking for the source code of the first primitive step in this
pipeline, first look at the primitive info in that step and get its
``id``:

.. code::

   "primitive": {
     "id": "4b42ce1e-9b98-4a25-b68e-fad13311eb65",
     "version": "0.3.0",
     "python_path": "d3m.primitives.data_transformation.dataset_to_dataframe.Common",
     "name": "Extract a DataFrame from a Dataset"
   },

Then, run this:

.. code:: shell

   git clone https://gitlab.com/datadrivendiscovery/common-primitives.git
   cd common-primitives
   grep -r 4b42ce1e-9b98-4a25-b68e-fad13311eb65 . | grep -F .py

However, this series of commands assumes that you know exactly which
specific repository is the primitive's source code located in (the ``git
clone`` command). Since this is probably not the case for an arbitrarily
given primitive, there is a method on how to find out the repository URL
of any primitive, and it requires using a d3m Docker image, which is
described in the next section.

Setting up a local d3m environment
----------------------------------

In order to run a pipeline, you must have a Python environment where the
d3m core package is installed, as well as the packages of the primitives
installed as well. While it is possible to setup a Python virtual
environment and install the packages them through ``pip``, in this
tutorial, we're going to use the d3m Docker images instead (in many
cases, even beyond this tutorial, this will save you a lot of time and
effort trying to find the any missing primitive packages, manually
installing them, and troubleshooting installation errors). So, make sure
`Docker <https://docs.docker.com/>`__ is installed in your system.

You can find the list of D3M docker images `here <https://docs.datadrivendiscovery.org/docker.html>`__.
The one we're going to use in this tutorial is the v2020.1.9
primitives image (feel free to use whatever the latest one instead
though - just modify the ``v2020.1.9`` part accordingly):

.. code:: shell

   docker pull registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9

Once you have downloaded the image, we can finally run the d3m package
(and hence run a pipeline). Before running a pipeline though, let's
first try to get a list of what primitives are installed in the image's
Python environment:

.. code:: shell

   docker run --rm registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 python3 -m d3m index search

You should get a big list of primitives. All of the known primitives to
D3M should be there.

You can also run the docker container in interactive mode (to run
commands as if you have logged into the container machine provides) by
using the ``-it`` option:

.. code:: shell

   docker run --rm -it registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9

The previous section mentions a method of determining where the source
code of an arbitrarily given primitive can be found. We can do this
using the d3m python package within a d3m docker container. First get the
``python_path`` of the primitive step (see the JSON snippet above of the
primitive's info from the pipeline). Then, run this command:

.. code:: shell

   docker run --rm registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 python3 -m d3m index describe d3m.primitives.data_transformation.dataset_to_dataframe.Common

Near the top of the huge JSON string describing the primitive, you'll see
``"source"``, and inside it, ``"uris"``. To help read the JSON, you can use
the ``jq`` utility:

.. code:: shell

   docker run --rm -it registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9
   python3 -m d3m index describe d3m.primitives.data_transformation.dataset_to_dataframe.Common | jq .source.uris

This should give the URI of the git repo where the source code of that primitive can be found. Also, You
can also substitute the primitive ``id`` for the ``python_path`` in that
command, but the command usually returns a result faster if you provide
the ``python_path``. Note also that you can only do this for primitives
that have been submitted for a particular image (primitives that are
contained in the `primitives index repo`_).

It can be obscure at first how to use the d3m python package, but you can
always access the help string for each d3m command at every level of the
command chain by using the ``-h`` flag. This is useful especially for
the getting a list of all the possible arguments for the ``runtime``
module.

.. code:: shell

   docker run --rm registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 python3 -m d3m -h
   docker run --rm registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 python3 -m d3m index -h
   docker run --rm registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 python3 -m d3m runtime -h
   docker run --rm registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 python3 -m d3m runtime fit-score -h

One last point before we try running a pipeline. The docker container
must be able to access the dataset location and the pipeline location
from the host filesystem. We can do this by `bind-mounting
<https://docs.docker.com/storage/bind-mounts/>`__ a host directory that
contains both the ``datasets`` repo and the ``primitives`` index repo to
a container directory. Git clone these repos, and also make another empty directory called
``pipeline-outputs``. Now, if your directory structure looks like this::

   /home/foo/d3m
   ├── datasets
   ├── pipeline-outputs
   └── primitives

Then you'll want to bind-mount ``/home/foo/d3m`` to a directory in the
container, say ``/mnt/d3m``. You can specify this mapping in the docker
command itself:

.. code:: shell

   docker run \
       --rm \
       -v /home/foo/d3m:/mnt/d3m \
       registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10 \
       ls /mnt/d3m

If you're reading this tutorial from a text editor, it might be a good
idea at this point to find and replace ``/home/foo/d3m`` with the actual
path in your system where the ``datasets``, ``pipeline-outputs``, and
``primitives`` directories are all located. This will make it easier for
you to just copy and paste the commands from here on out, instead of
changing the faux path every time.

.. _running-example-pipeline:

Running an example pipeline
---------------------------

At this point, let's try running a pipeline. Again, we're going to run
the example pipeline that comes with
``d3m.primitives.classification.logistic_regression.SKlearn``. There are
two ways to run a pipeline: by specifying all the necessary paths of the
dataset, or by specifying and using a pipeline run file. Let's
make sure first though that the dataset is available, as described in the
next subsection.

.. _preparing-dataset:

Preparing the dataset
~~~~~~~~~~~~~~~~~~~~~

Towards the end of the previous section, you were asked to git clone the
``datasets`` repo to your machine. Most likely, you might have
accomplished that like this:

.. code:: shell

   git clone https://datasets.datadrivendiscovery.org/d3m/datasets.git

But unless you had `git LFS <https://github.com/git-lfs/git-lfs>`__
installed, the entire contents of the repo might not have been really
installed.

The repo is organized such that all files larger than 100
KB is stored in git LFS. Thus, if you cloned without git LFS installed, you
most likely have to do a one-time extra step before you can use a dataset, as
some files of that dataset that are over 100 KB will not have the actual
data in them (although they will still exist as files in the cloned
repo). This is true even for the dataset that we will use in this
exercise, ``185_baseball``. To verify this, open this file in a text
editor::

   datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_dataset/tables/learningData.csv

Then, see if it contains text similar to this::

   version https://git-lfs.github.com/spec/v1
   oid sha256:931943cc4a675ee3f46be945becb47f53e4297ec3e470c4e3e1f1db66ad3b8d6
   size 131187

If it does, then this dataset has not yet been fully downloaded from git
LFS (but if it looks like a normal CSV file, then you can skip the rest
of this subsection and move on). To download this dataset, simply run
this command inside the ``datasets`` directory:

.. code:: shell

   git lfs pull -I training_datasets/seed_datasets_archive/185_baseball/

Inspect the file again, and you should see that it looks like a normal
CSV file now.

In general, if you don't know which specific dataset does a certain
example pipeline in the ``primitives`` repo uses, inspect the pipeline
run output file of that primitive (whose file path is similar to that of
the pipeline JSON file, as described in the :ref:`overview-of-primitives-and-pipelines` section, but
instead of going to ``pipelines``, go to ``pipeline_runs``). The
pipeline run is initially gzipped in the ``primitives`` repo, so
decompress it first. Then open up the actual .yml file, look at
``datasets``, and under it should be ``id``. If you do that for the
example pipeline run of the SKlearn logistic regression primitive
that we're looking at for this exercise, you'll find that the dataset id
is ``185_baseball_dataset``. The name of the main dataset directory is this string,
without the ``_dataset`` part.

Now, let's actually run the pipeline using the two ways mentioned
earlier.

Specifying all the necessary paths of a dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use this if there is no existing pipeline run yet for a
pipeline, or if you want to manually specify the dataset path (set the
paths for ``--problem``, ``--input``, ``--test-input``, ``--score-input``, ``--pipeline`` to your target dataset
location).

Remember to change the bind mount paths as appropriate for your system
(specified by ``-v``).

.. code:: shell

   docker run \
       --rm \
       -v /home/foo/d3m:/mnt/d3m \
       registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
       python3 -m d3m \
           runtime \
           fit-score \
               --problem /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_problem/problemDoc.json \
               --input /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json \
               --test-input /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/TEST/dataset_TEST/datasetDoc.json \
               --score-input /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/SCORE/dataset_TEST/datasetDoc.json \
               --pipeline /mnt/d3m/primitives/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipelines/862df0a2-2f87-450d-a6bd-24e9269a8ba6.json \
               --output /mnt/d3m/pipeline-outputs/predictions.csv \
               --output-run /mnt/d3m/pipeline-outputs/run.yml

The score is displayed after the pipeline run. The output predictions
will be stored on the path specified by ``--output``, and information about
the pipeline run is stored in the path specified by ``--output-run``.

Again, you can use the ``-h`` flag on ``fit-score`` to access the help
string and read about the different arguments, as described earlier.

If you get a python error that complains about missing columns, or
something that looks like this::

   ValueError: Mismatch between column name in data 'version https://git-lfs.github.com/spec/v1' and column name in metadata 'd3mIndex'.

Chances are that the ``185_baseball`` dataset has not yet been
downloaded through git LFS. See the :ref:`previous subsection
<preparing-dataset>` for details on how to verify and do this.

Using a pipeline run file
~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of specifying all the specific dataset paths, you can also use
an existing pipeline run to essentially "re-run" a previous run
of the pipeline:

.. code:: shell

   docker run \
       --rm \
       -v /home/foo/d3m:/mnt/d3m \
       registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
       python3 -m d3m \
           --pipelines-path /mnt/d3m/primitives/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipelines \
           runtime \
               --datasets /mnt/d3m/datasets \
           fit-score \
               --input-run /mnt/d3m/primitives/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipeline_runs/pipeline_run.yml.gz \
               --output /mnt/d3m/pipeline-outputs/predictions.csv \
               --output-run /mnt/d3m/pipeline-outputs/run.yml

In this case, ``--input-run`` is the pipeline run file that this pipeline
will re-run, and ``---output-run`` is the new pipeline run file that will be
generated.

Note that if you choose ``fit-score`` for the d3m runtime option, the
pipeline actually runs in two phases: fit, and produce. You can verify
this by searching for ``phase`` in the pipeline run file.

Lastly, if you want to run multiple commands in the docker container,
simply chain your commands with ``&&`` and wrap them double quotes
(``"``) for ``bash -c``. As an example:

.. code:: shell

   docker run \
       --rm \
       -v /home/foo/d3m:/mnt/d3m \
       registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
       /bin/bash -c \
           "python3 -m d3m \
               --pipelines-path /mnt/d3m/primitives/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipelines \
               runtime \
                   --datasets /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball \
               fit-score \
                   --input-run /mnt/d3m/primitives/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipeline_runs/pipeline_run.yml \
                   --output /mnt/d3m/pipeline-outputs/predictions.csv \
                   --output-run /mnt/d3m/pipeline-outputs/run.yml && \
           head /mnt/d3m/pipeline-outputs/predictions.csv"

Writing a new primitive
-----------------------

Let's now try to write a very simple new primitive - one that simply
passes whatever input data it receives from the previous step to the
next step in the pipeline. Let's call this primitive "Passthrough".

We will use this `skeleton primitive repo
<https://gitlab.com/datadrivendiscovery/docs-quickstart>`__
as a starting point
for this exercise. A d3m primitive repo does not have to follow the
exact same directory structure as this, but this is a good structure to
start with, at least. git clone the repo into ``docs-quickstart`` at the same place
where the other repos that we have used earlier are located
(``datasets``, ``pipeline-outputs``, ``primitives``).

Alternatively, you can also use the `test primitives
<https://gitlab.com/datadrivendiscovery/tests-data/tree/master/primitives>`__
as a model/starting point. ``test_primitives/null.py`` is essentially
the same primitive that we are trying to write.

.. _primitive-source-code:

Primitive source code
~~~~~~~~~~~~~~~~~~~~~

In the ``docs-quickstart`` directory, open
``quickstart_primitives/sample_primitive1/input_to_output.py``. The first
important thing to change here is the primitive metadata, which are the
first objects defined under the ``InputToOutputPrimitive`` class. Modify the
following fields (unless otherwise noted, the values you put in must be
strings):

- ``id``: The primitive's UUID v4 number/identifier. To generate one,
  you can run simply run this simple inline Python command:

  .. code:: shell

     python3 -c "import uuid; print(uuid.uuid4())"

- ``version``: You can use semantic versioning for this or another style
  of versioning. Write ``"0.1.0"`` for this exercise. You should bump
  the version of the primitive at least every time public interfaces
  of the primitive change (e.g. hyper-parameters).

- ``name``: The primitive's name. Write ``"Passthrough primitive"`` for
  this exercise.

- ``description``: A short description of the primitive. Write ``"A
  primitive which directly outputs the input."`` for this exercise.

- ``python_path``: This follows this format::

     d3m.primitives.<primitive family>.<primitive name>.<kind>

  Primitive families can be found in the `d3m metadata page
  <https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.primitive_family>`__
  (wait a few seconds for the page to load completely), and primitive
  names can be found in the `d3m core package source code
  <https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/primitive_names.py>`__.
  The last segment can be used to attribute the primitive to the author and/or
  describe in which way it is different from other primitives with same
  primitive family and primitive name, e.g., a different implementation with different
  trade-offs.

  For this exercise, write
  ``"d3m.primitives.operator.input_to_output.Quickstart"``. Note that
  ``input_to_output`` is not currently registered as a standard primitive name
  and using it will produce a warning. For primitives you intent on publishing
  make a merge request to the d3m core package to add any primitive names
  you need.

- ``primitive_family``: This must be the same as used for ``python_path``,
  as enumeration value. You can use a string or Python enumeration value.
  Add this import statement (if not there already):

  .. code:: python

     from d3m.metadata import base as metadata_base

  Then write ``metadata_base.PrimitiveFamily.OPERATOR`` (as
  a value, not a string, so do not put quotation marks) as the value of
  this field.

- ``algorithm_types``: Algorithm type(s) that the primitive implements.
  This can be multiple values in an array. Values can be chosen from
  the `d3m metadata page
  <https://metadata.datadrivendiscovery.org/devel/?definitions#definitions.algorithm_types>`__
  as well.
  Write ``[metadata_base.PrimitiveAlgorithmType.IDENTITY_FUNCTION]``
  here for this exercise (as a list that contains one element, not a
  string).

- ``source``: General info about the author of this primitive. ``name``
  is usually the name of the person or the team that wrote this
  primitive. ``contact`` is a ``mailto`` URI to the email address of
  whoever one should contact about this primitive. ``uris`` are usually
  the git clone URL of the repo, and you can also add the URL of the
  source file of this primitive.

  Write these for the exercise:

  .. code:: python

     "name": "My Name",
     "contact": "mailto:myname@example.com",
     "uris": ["https://gitlab.com/datadrivendiscovery/docs-quickstart.git"],

- ``keywords``: Key words for what this primitive is or does. Write
  ``["passthrough"]``.

- ``installation``: Information about how to install this primitive. Add
  these import statements first:

  .. code:: python

     import os.path
     from d3m import utils

  Then replace the ``installation`` entry with this:

  .. code:: python

     "installation": [{
         "type": metadata_base.PrimitiveInstallationType.PIP,
         "package_uri": "git+https://gitlab.com/datadrivendiscovery/docs-quickstart@{git_commit}#egg=quickstart_primitives".format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__))
         ),
     }],

  In general, for your own actual primitives, you might only need to
  substitute the git repo URL here as well as the python egg name.

Next, let's take a look at the ``produce`` method. You can see that it
simply makes a new dataframe out of the input data, and returns it as
the output. To see for ourselves though that our primitive (and thus
this ``produce`` method) gets called during the pipeline run, let's add
a log statement here. The ``produce`` method should now look something
like this:

.. code:: python

   def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
       self.logger.warning('Hi, InputToOutputPrimitive.produce was called!')
       return base.CallResult(value=inputs)

Note that this is simply an example primitive that is intentionally
simple for the purposes of this tutorial. It does not necessarily model
a well-written primitive, by any means. For guidelines on how to write a
good primitive, take a look at the :ref:`primitive-good-citizen`.

setup.py
~~~~~~~~

Next, we fill in the necessary information in ``setup.py`` so that
``pip`` can correctly install our primitive in our local d3m
environment. Open ``setup.py`` (in the project root), and modify the
following fields:

- ``name``: Same as the egg name you used in ``package_uri``

- ``version``: Same as the primitive metadata's ``version``

- ``description``: Same as the primitive metadata's ``description``,
  or a description of all primitives if there are multiple primitives
  in the package you are making

- ``author``: Same as the primitive metadata's ``suorce.name``

- ``url``: Same as main URL in the primitive metadata's
  ``source.uris``

- ``packages``: This is an array of the python packages that this
  primitive repo contains. You can use the ``find_packages`` helper:

  .. code:: python

     packages=find_packages(exclude=['pipelines']),

- ``keywords``: A list of keywords. Important standard keyword is
  ``d3m_primitive`` which makes all primitives discoverable on PyPi

- ``install_requires``: This is an array of the python package
  dependencies of the primitives contained in this repo. Our primitive
  needs nothing except the d3m core package (and the
  ``common-primitives`` package too for testing, but this is not a
  package dependency), so write this as the value of this field:
  ``['d3m']``

- ``entry_points``: This is how the d3m runtime maps your primitives'
  d3m python paths to the your repo's local python paths. For this
  exercise, it should look like this:

  .. code:: python

     entry_points={
         'd3m.primitives': [
             'operator.input_to_output.Quickstart = quickstart_primitives.sample_primitive1:InputToOutputPrimitive',
         ],
     }

That's it for this file. Briefly review it for any possible syntax
errors.

Primitive unit tests
~~~~~~~~~~~~~~~~~~~~

Let's now make a python test for this primitive, which in this case will
just assert whether the input dataframe to the primitive equals the
output dataframe. Make a new file called ``test_input_to_output.py``
inside ``quickstart_primitives/sample_primitive1`` (the same directory as
``input_to_output.py``), and write this as its contents:

.. code:: python

   import unittest
   import os

   from d3m import container
   from common_primitives import dataset_to_dataframe
   from input_to_output import InputToOutputPrimitive


   class InputToOutputTestCase(unittest.TestCase):
       def test_output_equals_input(self):
           dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tests-data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json'))

           dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

           dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
           dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
           dataframe = dataframe_primitive.produce(inputs=dataset).value

           i2o_hyperparams_class = InputToOutputPrimitive.metadata.get_hyperparams()
           i2o_primitive = InputToOutputPrimitive(hyperparams=dataframe_hyperparams_class.defaults())
           output = i2o_primitive.produce(inputs=dataframe).value

           self.assertTrue(output.equals(dataframe))


   if __name__ == '__main__':
       unittest.main()

For the dataset that this test uses, add as git submodule the `d3m tests-data <https://gitlab.com/datadrivendiscovery/tests-data>`__
repository at the root of the ``docs-quickstart`` repository.
Then let's install this new primitive to the Docker image's d3m environment, and
run this test using the command below:

.. code:: shell

   docker run \
       --rm \
       -v /home/foo/d3m:/mnt/d3m \
       registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
       /bin/bash -c \
           "pip3 install -e /mnt/d3m/docs-quickstart && \
           cd /mnt/d3m/docs-quickstart/quickstart_primitives/sample_primitive1 && \
           python3 test_input_to_output.py"

You should see a log statement like this, as well as the python unittest
pass message::

   Hi, InputToOutputPrimitive.produce was called!
   .
   ----------------------------------------------------------------------
   Ran 1 test in 0.011s

Using this primitive in a pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having seen the primitive test pass, we can now confidently include this
primitive in a pipeline. Let's take the same pipeline that we ran :ref:`before <running-example-pipeline>`
(the sklearn logistic regression's example pipeline),
and add a step using this primitive.

In the root directory of your repository, create these directories:
``pipelines/operator.input_to_output.Quickstart``. Then, from the d3m
``primitives`` repo, copy the JSON pipeline description file from
``primitives/v2020.1.9/JPL/d3m.primitives.classification.logistic_regression.SKlearn/2019.11.13/pipelines``
into the directory we just created. Open this file, and replace the
``id`` (generate another UUID v4 number using the inline python command
earlier, different from the primitive ``id``), as well as the created
timestamp using this inline python command (add ``Z`` at the end of the
generated timestamp)::

   python3 -c "import time; import datetime; \
   print(datetime.datetime.fromtimestamp(time.time()).isoformat())"

You can rename the json file too using the new pipeline ``id``.

Next, change the output step number (shown below, ``"steps.4.produce"``)
to be one more than the current number (at the time of this writing, it
is ``4``, so in this case, change it to ``5``):

.. code:: json

   "outputs": [
     {
       "data": "steps.5.produce",
       "name": "output predictions"
     }
   ],

Then, find the step that contains the
``d3m.primitives.classification.logistic_regression.SKlearn`` primitive
(search for this string in the file), and right above it, add the
following JSON object. Remember to change ``primitive.id`` to the
primitive's id that you generated in the earlier :ref:`primitive-source-code` subsection.

.. code:: json

   {
     "type": "PRIMITIVE",
     "primitive": {
       "id": "30d5f2fa-4394-4e46-9857-2029ec9ed0e0",
       "version": "0.1.0",
       "python_path": "d3m.primitives.operator.input_to_output.Quickstart",
       "name": "Passthrough primitive"
     },
     "arguments": {
       "inputs": {
         "type": "CONTAINER",
         "data": "steps.2.produce"
       }
     },
     "outputs": [
       {
         "id": "produce"
       }
     ]
   },

Make sure that the step number (``"steps.N.produce"``) in
``arguments.inputs.data`` is correct (one greater than the previous step
and one less than the next step). Do this as well for the succeeding
steps, with the following caveats:

- For ``d3m.primitives.classification.logistic_regression.SKlearn``,
  increment the step number both for ``arguments.inputs.data`` and
  ``arguments.outputs.data`` (at the time of this writing, the number
  should be changed to ``3``).
- For
  ``d3m.primitives.data_transformation.construct_predictions.Common``,
  increment the step number for ``arguments.inputs.data`` (at the time
  of this writing, the number should be changed to ``4``), but do not
  change the one for ``arguments.reference.data`` (the value should
  stay as ``"steps.0.produce"``)

Generally, you can also programmatically generate a pipeline, as
described in the :ref:`pipeline-description-example`.

Now we can finally run this pipeline that uses our new primitive. In the
command below, modify the pipeline JSON filename in the ``-p`` argument
to match the filename of your pipeline file (if you changed it to the
new pipeline id that you generated).

.. code:: shell

   docker run \
       --rm \
       -v /home/foo/d3m:/mnt/d3m \
       registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9 \
       /bin/bash -c \
           "pip3 install -e /mnt/d3m/docs-quickstart && \
           python3 -m d3m \
               runtime \
               fit-score \
                   --problem /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/185_baseball_problem/problemDoc.json \
                   --input /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json \
                   --test-input /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/TEST/dataset_TEST/datasetDoc.json \
                   --score-input /mnt/d3m/datasets/training_datasets/seed_datasets_archive/185_baseball/SCORE/dataset_TEST/datasetDoc.json \
                   --pipeline /mnt/d3m/docs-quickstart/pipelines/operator.input_to_output.Quickstart/0f290525-3fec-44f7-ab93-bd778747b91e.json \
                   --output /mnt/d3m/pipeline-outputs/predictions_new.csv \
                   --output-run /mnt/d3m/pipeline-outputs/run_new.yml"

In the output, you should see the log statement as a warning,
before the score is shown (similar to the text below)::

   ...
   WARNING:d3m.primitives.operator.input_to_output.Quickstart:Hi, InputToOutputPrimitive.produce was called!
   ...
   metric,value,normalized,randomSeed
   F1_MACRO,0.31696136214800263,0.31696136214800263,0

Verify that the old and new ``predictions.csv`` in ``pipeline-outputs``
are the same (you can use ``diff``), as well as the scores in the old
and new ``run.yml`` files (search for ``scores`` in the files).

Beyond this tutorial
--------------------

Congratulations! You just built your own primitive and you were able to
use it in a d3m pipeline!

Normally, when you build your own primitives, you would proceed to
validating the primitives to be included in the d3m index of all known
primitives. See the `primitives repo README
<https://gitlab.com/datadrivendiscovery/primitives#adding-a-primitive>`__
on details on how to do this.
