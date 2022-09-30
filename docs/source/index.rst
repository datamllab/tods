.. Time Series Outlier Detection System documentation master file, created by
   sphinx-quickstart on Wed Sep  9 22:52:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TODS's documentation!
================================================================
.. image:: img/tods_logo.png
   :width: 400

TODS is a full-stack automated machine learning system for outlier detection on multivariate time-series data. TODS provides exahaustive modules for building machine learning-based outlier detection systems including: data processing, time series processing, feature analysis (extraction), detection algorithms, and reinforcement module. The functionalities provided via these modules including: data preprocessing for general purposes, time series data smoothing/transformation, extracting features from time/frequency domains, various detection algorithms, and involving human expertises to calibrate the system. Three common outlier detection scenarios on time-series data can be performed: point-wise detection (time points as outliers), pattern-wise detection (subsequences as outliers), and system-wise detection (sets of time series as outliers), and wide-range of corresponding algorithms are provided in TODS. This package is developed by `DATA Lab @ Rice University <https://people.engr.tamu.edu/xiahu/index.html>`__.

TODS is featured for:

* **Full Stack Machine Learning System** which supports exhaustive components from preprocessings, feature extraction, detection algorithms and also human-in-the loop interface. 

* **Wide-range of Algorithms**, including all of the point-wise detection algorithms supported by `PyOD <https://github.com/yzhao062/pyod>`__, state-of-the-art pattern-wise (collective) detection algorithms such as `DeepLog <https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf>`__, `Telemanon <https://arxiv.org/pdf/1802.04431.pdf>`__, and also various ensemble algorithms for performing system-wise detection.

* **Automated Machine Learning** aims on providing knowledge-free process that construct optimal pipeline based on the given data by automatically searching the best combination from all of the existing modules.

Installation
------------
This package works with **Python 3.6** and pip 19+. You need to have the following packages installed on the system (for Debian/Ubuntu):
(if you are in China and Github is slow, you can use the mirror in `Gitee <https://gitee.com/daochenzha/tods>`__):
::
    sudo apt-get install libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg

Then execute ``python setup.py install``, the script will then install all of the packges to build up TODS.



.. toctree::
   :maxdepth: 4
   :caption: Contents:

Examples
--------
Examples are available in `examples <https://github.com/datamllab/tods/tree/master/examples>`__. For basic usage, you can evaluate a pipeline on a given datasets. Here, we provide an example to load our default pipeline and evaluate it on a subset of yahoo dataset.

.. code:: python

    import pandas as pd

    from tods import schemas as schemas_utils
    from tods.utils import generate_dataset_problem, evaluate_pipeline

    table_path = 'datasets/yahoo_sub_5.csv'
    target_index = 6 # what column is the target
    metric = 'F1_MACRO' # F1 on both label 0 and 1

    # Read data and generate dataset and problem
    df = pd.read_csv(table_path)
    dataset, problem_description = generate_dataset_problem(df, target_index)

    # Load the default pipeline
    pipeline = schemas_utils.load_default_pipeline()

    # Run the pipeline
    pipeline_result = evaluate_pipeline(dataset, pipeline, metric)


We also provide AutoML support to help you automatically find a good pipeline for a your data.


.. code:: python

    import pandas as pd

    from axolotl.backend.simple import SimpleRunner

    from tods.utils import generate_dataset_problem
    from tods.search import BruteForceSearch

    # Some information
    #table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_GOOG.csv' # The path of the dataset
    #target_index = 2 # what column is the target

    table_path = 'datasets/yahoo_sub_5.csv'
    target_index = 6 # what column is the target
    time_limit = 30 # How many seconds you wanna search
    metric = 'F1_MACRO' # F1 on both label 0 and 1

    # Read data and generate dataset and problem
    df = pd.read_csv(table_path)
    dataset= generate_dataset(df, target_index=target_index)
    problem_description = generate_problem(dataset, metric)

    # Start backend
    backend = SimpleRunner(random_seed=0)

    # Start search algorithm
    search = BruteForceSearch(problem_description=problem_description, backend=backend)

    # Find the best pipeline
    best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=time_limit)
    best_pipeline = best_runtime.pipeline
    best_output = best_pipeline_result.output

    # Evaluate the best pipeline
    best_scores = search.evaluate(best_pipeline).scores


.. rlcard documentation master file, created by
   sphinx-quickstart on Thu Sep  5 18:45:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :glob:
   :caption: Documentation:

   overview
   getting_started
   PythonPathCheatSheet


.. toctree::
   :glob:
   :caption: API Documents:

   tods.data_processing
   tods.timeseries_processing
   tods.feature_analysis
   tods.detection_algorithm
   tods.reinforcement



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
