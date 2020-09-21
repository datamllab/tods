
# TODS: Automated Time-series Outlie Detection System
<img width="500" src="./docs/img/tods_logo.png" alt="Logo" />

[![Build Status](https://travis-ci.org/datamllab/tods.svg?branch=master)](https://travis-ci.org/datamllab/tods)

TODS is a full-stack automated machine learning system for outlier detection on multivariate time-series data. TODS provides exahaustive modules for building machine learning-based outlier detection systems including: data processing, time series processing, feature analysis (extraction), detection algorithms, and reinforcement module. The functionalities provided via these modules including: data preprocessing for general purposes, time series data smoothing/transformation, extracting features from time/frequency domains, various detection algorithms, and involving human expertises to calibrate the system. Three common outlier detection scenarios on time-series data can be performed: point-wise detection (time points as outliers), pattern-wise detection (subsequences as outliers), and system-wise detection (sets of time series as outliers), and wide-range of corresponding algorithms are provided in TODS. This package is developed by [DATA Lab @ Texas A&M University](https://people.engr.tamu.edu/xiahu/index.html).

TODS is featured for:
* **Full Sack Machine Learning System** which supports exhaustive components from preprocessings, feature extraction, detection algorithms and also human-in-the loop interface. 

* **Wide-range of Algorithms**, including all of the point-wise detection algorithms supported by [PyOD](https://github.com/yzhao062/pyod), state-of-the-art pattern-wise (collective) detection algorithms such as [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf), [Telemanon](https://arxiv.org/pdf/1802.04431.pdf), and also various ensemble algorithms for performing system-wise detection.

* **Automated Machine Learning** aims on providing knowledge-free process that construct optimal pipeline based on the given data by automatically searching the best combination from all of the existing modules.

## Resources
* API Documentations: [http://tods-doc.github.io](http://tods-doc.github.io)

## Installation

This package works with **Python 3.6** and pip 19+. You need to have the following packages installed on the system (for Debian/Ubuntu):
```
sudo apt-get install libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg
```

Clone the repository:
```
git clone https://github.com/datamllab/tods.git
```
Install locally with `pip`:
```
cd tods
pip install -e .
```

# Examples
Examples are available in [/examples](examples/). For basic usage, you can evaluate a pipeline on a given datasets. Here, we provide an example to load our default pipeline and evaluate it on a subset of yahoo dataset.
```python
import pandas as pd

from tods import schemas as schemas_utils
from tods import generate_dataset, evaluate_pipeline

table_path = 'datasets/yahoo_sub_5.csv'
target_index = 6 # what column is the target
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = schemas_utils.load_default_pipeline()

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)
```
We also provide AutoML support to help you automatically find a good pipeline for a your data.
```python
import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods import generate_dataset, generate_problem
from tods.searcher import BruteForceSearch

# Some information
table_path = 'datasets/yahoo_sub_5.csv'
target_index = 6 # what column is the target
time_limit = 30 # How many seconds you wanna search
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index=target_index)
problem_description = generate_problem(dataset, metric)

# Start backend
backend = SimpleRunner(random_seed=0)

# Start search algorithm
search = BruteForceSearch(problem_description=problem_description,
                          backend=backend)

# Find the best pipeline
best_runtime, best_pipeline_result = search.search_fit(input_data=[dataset], time_limit=time_limit)
best_pipeline = best_runtime.pipeline
best_output = best_pipeline_result.output

# Evaluate the best pipeline
best_scores = search.evaluate(best_pipeline).scores
```
# Acknowledgement
We gratefully acknowledge the Data Driven Discovery of Models (D3M) program of the Defense Advanced Research Projects Agency (DARPA)


