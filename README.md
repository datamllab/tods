# Automated Time-series Outlie Detection System
This is a time-seried outlier detection system with automate machin learning.

## Axolotl
Running pre-defined pipeline
```
python examples/build_AutoEncoder_pipeline.py
python examples/run_predefined_pipeline.py
```

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

# Dataset
Datasets are located in `datasets/anomaly`. `raw_data` is the raw time series data. `transform.py` is script to transform the raw data to D3M format. `template` includes some templates for generating D3M data. If you run `transform.py`, the script will load the raw `kpi` data and create a folder named `kpi` in D3M format.

The generated csv file will have the following columns: `d3mIndex`, `timestamp`, `value`, `'ground_truth`. In the example kpi dataset, there is only one value. For other datasets there could be multiple values. The goal of the pipline is to predict the `ground_truth` based on `timestamp` and the value(s).

There is a nice script to check whether the dataset is in the right format. Run
```
python3 datasets/validate.py datasets/anomaly/kpi/
```
The expected output is as follows:
```
Validating problem '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/SCORE/problem_TEST/problemDoc.json'.
Validating dataset '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/SCORE/dataset_TEST/datasetDoc.json'.
Validating problem '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/kpi_problem/problemDoc.json'.
Validating problem '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/TEST/problem_TEST/problemDoc.json'.
Validating dataset '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/TEST/dataset_TEST/datasetDoc.json'.
Validating dataset '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/kpi_dataset/datasetDoc.json'.
Validating dataset '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/TRAIN/dataset_TRAIN/datasetDoc.json'.
Validating problem '/home/grads/d/daochen/tods/tods/datasets/anomaly/kpi/TRAIN/problem_TRAIN/problemDoc.json'.
Validating all datasets and problems.
There are no errors.
```
Of course, you can also create other datasets with `transform.py`. But for now, we can focus on this example dataset since other datasets are usually in the same format.

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

# How to add a new primitive

For new primitives, put them in `/anomaly_pritives`. There is an example for isolation forest (however, this is essentially a RandomForest, although the name is IsolationForest. We need more efforts to change it to real IsolationForest).

In addition to add a new file, you need to register the promitive in `anomaly-primitives/setup.py` and rerun pip install.

Use the following command to check whether your new primitives are registered:
```
python3 -m d3m index search
```

Test the new primitives:
```
python3 examples/build_iforest_pipline.py
```

# Template for meta-data in primitives

*   `__author__`: `DATA Lab at Texas A&M University`
*   `name`: Just a name. Name your primitive with a few words
*   `python_path`: This path should have **5** segments. The first two segments should be `d3m.primitives`. The third segment shoulb be `anomaly_detection`, `data_preprocessing` or `feature_construction` (it should match `primitive_family`). The fourth segment should be your algorithm name, e.g., `isolation_forest`. Note that this name should also be added to [this file](d3m/d3m/metadata/primitive_names.py). The last segment should be one of `Preprocessing`, `Feature`, `Algorithm` (for now).
*   `source`: `name` should be `DATA Lab at Texas A&M University`, `contact` should be `mailto:khlai037@tamu.edu`, `uris` should have `https://gitlab.com/lhenry15/tods.git` and the path your py file.
*   `algorithms_types`: Name the primitive by your self and add it to [here](d3m/d3m/metadata/schemas/v0/definitions.json#L1957). **Then reinstall d3m.** Fill this field with `metadata_base.PrimitiveAlgorithmType.YOUR_NAME`
*   `primitive_family`: For preprocessing primitives, use `metadata_base.PrimitiveFamily.DATA_PREPROCESSING`. For feature analysis primitives, use `metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION`. For anomaly detection primitives, use `metadata_base.PrimitiveFamily.ANOMALY_DETECTION`.
*   `id`: Randomly generate one with `import uuid; uuid.uuid4()`
*   `hyperparameters_to_tune`: Specify what hyperparameters can be tuned in your primitive
*   `version`: `0.0.1`

Notes:

1. `installation` is not required. We remove it.

2. Try to reinstall everything if it does not work.

3. An example of fake Isolation Forest is [here](anomaly-primitives/anomaly_primitives/SKIsolationForest.py#L294)


## Resources of D3M

If you still have questions, you may refer to the following resources.

Dataset format [https://gitlab.com/datadrivendiscovery/data-supply](https://gitlab.com/datadrivendiscovery/data-supply)

Instructions for creating primitives [https://docs.datadrivendiscovery.org/v2020.1.9/interfaces.html](https://docs.datadrivendiscovery.org/v2020.1.9/interfaces.html)

We use a stable version of d3m core package at [https://gitlab.com/datadrivendiscovery/d3m/-/tree/v2020.1.9](https://gitlab.com/datadrivendiscovery/d3m/-/tree/v2020.1.9).

The documentation is at [https://docs.datadrivendiscovery.org/](https://docs.datadrivendiscovery.org/).

The core package documentation is at [https://docs.datadrivendiscovery.org/v2020.1.9/index.html](https://docs.datadrivendiscovery.org/v2020.1.9/index.html)

The common-primitives is v0.8.0 at [https://gitlab.com/datadrivendiscovery/common-primitives/-/tree/v0.8.0/common_primitives](https://gitlab.com/datadrivendiscovery/common-primitives/-/tree/v0.8.0/common_primitives)

The sklearn-wrap uses dist branch [https://gitlab.com/datadrivendiscovery/sklearn-wrap/-/tree/dist](https://gitlab.com/datadrivendiscovery/sklearn-wrap/-/tree/dist)

There are other primitives developed by many universities but are not used in this repo. See [https://gitlab.com/datadrivendiscovery/primitives](https://gitlab.com/datadrivendiscovery/primitives)
