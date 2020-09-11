# Time-series Outlie Detection System
TODS is a full-stack automated machine learning system for outlier detection on multivariate time-series data. TODS provides exahaustive modules for building machine learning-based outlier detection systems including: data processing, time series processing, feature analysis (extraction), detection algorithms, and reinforcement module. The functionalities provided via these modules including: data preprocessing for general purposes, time series data smoothing/transformation, extracting features from time/frequency domains, various detection algorithms, and involving human expertises to calibrate the system. Three common outlier detection scenarios on time-series data can be performed: point-wise detection (time points as outliers), pattern-wise detection (subsequences as outliers), and system-wise detection (sets of time series as outliers), and wide-range of corresponding algorithms are provided in TODS. This package is developed by [DATA Lab @ Texas A&M University](https://people.engr.tamu.edu/xiahu/index.html).

TODS is featured for:
* **Full Sack Machine Learning System** which supports exhaustive components from preprocessings, feature extraction, detection algorithms and also human-in-the loop interface. 

* **Wide-range of Algorithms**, including all of the point-wise detection algorithms supported by [PyOD](https://github.com/yzhao062/pyod), state-of-the-art pattern-wise (collective) detection algorithms such as [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf), [Telemanon](https://arxiv.org/pdf/1802.04431.pdf), and also various ensemble algorithms for performing system-wise detection.

* **Automated Machine Learning** aims on providing knowledge-free process that construct optimal pipeline based on the given data by automatically searching the best combination from all of the existing modules.


## Installation

This package works with **Python 3.6** and pip 19+. You need to have the following packages installed on the system (for Debian/Ubuntu):
```
sudo apt-get install libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg
```

Then run the script `install.sh`. The script witll install d3m core package with:
```
cd d3m
pip3 install -e .
cd ..
```
Then it installs common primitives (which will be used in the running examples):
```
cd common-primitives
pip3 install -e .
cd ..
```
And it installs sklearn wrapper with:
```
cd sklearn-wrap
pip3 install -r requirements.txt
pip3 install -e .
cd ..
```
It installs anomaly primitives (ours) by:
```
cd anomaly-primitives
pip3 install -r requirements.txt
pip3 install -e .
cd ..
```

There could be some missing dependencies that are not listed above. Try to fix it by yourself if you meet any.

# Examples
Examples are available in [/examples](examples/). For basic usage, you can evaluate a pipeline on a given datasets. Here, we provide an example to load our default pipeline and evaluate it on a subset of yahoo dataset.
```python
import pandas as pd

from tods import schemas as schemas_utils
from tods.utils import generate_dataset_problem, evaluate_pipeline

table_path = 'datasets/yahoo_sub_5.csv'
target_index = 6 # what column is the target
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset
time_limit = 30 # How many seconds you wanna search
#metric = 'F1' # F1 on label 1
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset, problem_description = generate_dataset_problem(df, target_index=target_index, metric=metric)

# Load the default pipeline
pipeline = schemas_utils.load_default_pipeline()

# Run the pipeline
pipeline_result = evaluate_pipeline(problem_description, dataset, pipeline)
```
We also provide AutoML support to help you automatically find a good pipeline for a your data.
```python
import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods.utils import generate_dataset_problem
from tods.search import BruteForceSearch

# Some information
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_GOOG.csv' # The path of the dataset
#target_index = 2 # what column is the target

table_path = 'datasets/yahoo_sub_5.csv'
target_index = 6 # what column is the target
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset
time_limit = 30 # How many seconds you wanna search
#metric = 'F1' # F1 on label 1
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset and problem
df = pd.read_csv(table_path)
dataset, problem_description = generate_dataset_problem(df, target_index=target_index, metric=metric)

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
```
# Example
In D3M, our goal is to provide a **solution** to a **problem** on a **dataset**. Here, solution is a pipline which consists of data processing, classifiers, etc.

Run the example to build the first pipline with
```
python3 examples/build_iforest_pipline.py
```
Note that we have not implemented iForest yet. This one is actually Random Forest. This will generate a file `pipline.yml`, which describes a pipline. We can run the pipeline on the example data in this repo as follows:
```
python3 -m d3m runtime fit-produce -p pipeline.yml -r datasets/anomaly/kpi/TRAIN/problem_TRAIN/problemDoc.json -i datasets/anomaly/kpi/TRAIN/dataset_TRAIN/datasetDoc.json -t datasets/anomaly/kpi/TEST/dataset_TEST/datasetDoc.json -o results.csv -O pipeline_run.yml
```
Another example on a subset of the sequences of Yahoo dataset is as follows:
```
python3 -m d3m runtime fit-produce -p pipeline.yml -r datasets/anomaly/yahoo_sub_5/TRAIN/problem_TRAIN/problemDoc.json -i datasets/anomaly/yahoo_sub_5/TRAIN/dataset_TRAIN/datasetDoc.json -t datasets/anomaly/yahoo_sub_5/TEST/dataset_TEST/datasetDoc.json -o results.csv -O pipeline_run.yml
```
The above commands will generate two files `results.csv` and `pipline_run.yml`

