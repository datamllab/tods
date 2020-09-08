Installation
------------

This package works with Python 3.6+ and pip 19+. You need to have the following
packages installed on the system (for Debian/Ubuntu):

-  ``libssl-dev``
-  ``libcurl4-openssl-dev``
-  ``libyaml-dev``

You can install latest stable version from `PyPI <https://pypi.org/>`__:

::

    $ pip3 install d3m

To install latest development version:

::

    $ pip3 install -e git+https://gitlab.com/datadrivendiscovery/d3m.git@devel#egg=d3m

When cloning a repository, clone it recursively to get also git
submodules:

::

    $ git clone --recursive https://gitlab.com/datadrivendiscovery/d3m.git

Testing
-------

To ensure consistent performance of the D3M package a test suite and performance benchmarks are ran in the CI pipeline after every commit.
If a commit fails tests or introduces significant performance regression the pipeline fails.

Running tests
~~~~~~~~~~~~~

To run the test suite locally run:

::

    $ ./run_tests.py

Running benchmarks
~~~~~~~~~~~~~~~~~~

If you want to run benchmarks locally you first need to install asv:

::

    $ pip install asv

then clone the D3M repository:

::

    $ git clone git@gitlab.com:datadrivendiscovery/d3m.git
    $ cd d3m/tests

and run the benchmarks on a set of git commits. The following command:

::

    asv continuous --config asv.conf.json -f 1.1 devel HEAD

will benchmarks changes between last commit to `devel` and latest commit to currently active feature branch.
Make sure the code you want to benchmark is commited into active git branch.

To inspect performance changes between last two commits in the active branch run:

::

    $ asv continuous --config asv.conf.json -f 1.1 HEAD
    · Creating environments
    · Discovering benchmarks
    ·· Uninstalling from virtualenv-py3.6
    ·· Installing a1bb2749 <asv_benchmarks> into virtualenv-py3.6.
    · Running 4 total benchmarks (2 commits * 1 environments * 2 benchmarks)
    [  0.00%] · For d3m commit 3759f7a7 <asv_benchmarks~1> (round 1/2):
    [  0.00%] ·· Building for virtualenv-py3.6.
    [  0.00%] ·· Benchmarking virtualenv-py3.6
    [ 12.50%] ··· Running (metadata.DatasetMetadata.time_update_0k--)..
    [ 25.00%] · For d3m commit a1bb2749 <asv_benchmarks> (round 1/2):
    [ 25.00%] ·· Building for virtualenv-py3.6.
    [ 25.00%] ·· Benchmarking virtualenv-py3.6
    [ 37.50%] ··· Running (metadata.DatasetMetadata.time_update_0k--)..
    [ 50.00%] · For d3m commit a1bb2749 <asv_benchmarks> (round 2/2):
    [ 50.00%] ·· Benchmarking virtualenv-py3.6
    [ 62.50%] ··· metadata.DatasetMetadata.time_update_0k               2.84±0.4ms
    [ 75.00%] ··· metadata.DatasetMetadata.time_update_1k                  174±4ms
    [ 75.00%] · For d3m commit 3759f7a7 <asv_benchmarks~1> (round 2/2):
    [ 75.00%] ·· Building for virtualenv-py3.6.
    [ 75.00%] ·· Benchmarking virtualenv-py3.6
    [ 87.50%] ··· metadata.DatasetMetadata.time_update_0k               5.59±0.5ms
    [100.00%] ··· metadata.DatasetMetadata.time_update_1k                 714±10ms
           before           after         ratio
         [3759f7a7]       [a1bb2749]
         <asv_benchmarks~1>       <asv_benchmarks>
    -      5.59±0.5ms       2.84±0.4ms     0.51  metadata.DatasetMetadata.time_update_0k
    -        714±10ms          174±4ms     0.24  metadata.DatasetMetadata.time_update_1k


During development, you can run a particular benchmark using the current environment and code by::

    $ asv dev --config asv.conf.json --bench 'metadata.DatasetToJsonStructure.time_to_json_structure.*'

For additional reference the following resources can be useful:

-  `Pandas performance test suite guide <http://pandas.pydata.org/pandas-docs/stable/development/contributing.html#running-the-performance-test-suite>__`
-  `Asv usage guide <https://asv.readthedocs.io/en/stable/using.html>__`
-  `Astropy benchmarks <https://github.com/astropy/astropy-benchmarks>__`
