# Revisiting Time Series Outlier Detection:Definitions and Benchmarks

This branch is the source code of  experiment part of our paper. We provide everything needed when running the experiments: Dataset, Dataset Generator, Pipeline json, Python script, runner and the result (in "./result") we get from the experiments.

## Resources
* Paper: Under review


## Datasets
To get the dataset, please go to "data/script" to run all of the python scripts. They will download and preprocess the data automatically into "data/" folder.


## Pipeline

This Pipeline json files are organized by different settings of algorithms. 

## Runner

To run a pipeline, you can generate your own pipeline json file from script.

```python
sh build_pipelines.sh
```

Then run the pipeline with run\_pipeline.py (Below is the example for running IForest on GECCO dataset)
```python
python run_pipeline.py --pipeline_path pipelines/simple/pyod_iforest_0.01.json --data_path ./data/water_quality.csv
```



Or you can directly use the pipelines we have generated in /pipelines with bash script:

```python
sh run.sh
```


## Cite this Work:
If you find this  work useful, you may cite this work:
```
@misc{lai2020tods,
    title={TODS: An Automated Time Series Outlier Detection System},
    author={Kwei-Harng Lai and Daochen Zha and Guanchu Wang and Junjie Xu and Yue Zhao and Devesh Kumar and Yile Chen and Purav Zumkhawaka and Minyang Wan and Diego Martinez and Xia Hu},
    year={2020},
    eprint={2009.09822},
    archivePrefix={arXiv},
    primaryClass={cs.DB}
}
```
*Please refer master branch of TODS for details of running pipelines.
