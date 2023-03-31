
# TODS: Automated Time-series Outlier Detection System 自动化时间序列异常检测系统
<img width="500" src="./docs/img/tods_logo.png" alt="Logo" />

[![Build Status](https://travis-ci.org/datamllab/tods.svg?branch=master)](https://travis-ci.org/datamllab/tods)

[English README](README.md)

TODS是一个全栈的自动化机器学习系统，主要针对多变量时间序列数据的异常检测。TODS提供了详尽的用于构建基于机器学习的异常检测系统的模块，它们包括：数据处理（data processing），时间序列处理（ time series processing），特征分析（feature analysis)，检测算法（detection algorithms），和强化模块（ reinforcement module）。这些模块所提供的功能包括常见的数据预处理、时间序列数据的平滑或变换，从时域或频域中抽取特征、多种多样的检测算法以及让人类专家来校准系统。该系统可以处理三种常见的时间序列异常检测场景：点的异常检测（异常是时间点）、模式的异常检测（异常是子序列）、系统的异常检测（异常是时间序列的集合）。TODS提供了一系列相应的算法。该包由 [DATA Lab @ Rice University](https://people.engr.tamu.edu/xiahu/index.html) 开发。

TODS具有如下特点：
* **全栈式机器学习系统**：支持从数据预处理、特征提取、到检测算法和人为规则每一个步骤并提供相应的接口。

* **广泛的算法支持**：包括[PyOD](https://github.com/yzhao062/pyod) 提供的点的异常检测算法、最先进的模式的异常检测算法（例如 [DeepLog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf), [Telemanon](https://arxiv.org/pdf/1802.04431.pdf) ），以及用于系统的异常检测的集合算法。

* **自动化的机器学习**：旨在提供无需专业知识的过程，通过自动搜索所有现有模块中的最佳组合，基于给定数据构造最优管道。

## 相关资源
* API文档: [http://tods-doc.github.io](http://tods-doc.github.io)
* 论文: [https://arxiv.org/abs/2009.09822](https://arxiv.org/abs/2009.09822)
* 相关项目：[AutoVideo: An Automated Video Action Recognition System](https://github.com/datamllab/autovideo)
* :loudspeaker: 想了解更多数据管道搜索吗? 请关注我们的 [data-centric AI survey](https://arxiv.org/abs/2303.10158) 和 [data-centric AI resources](https://github.com/daochenzha/data-centric-AI)!

## 引用该工作：
如何您觉得我们的工作有用，请引用该工作：
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

## 安装

这个包的运行环境是 **Python 3.6** 和pip 19+。对于Debian和Ubuntu的使用者，您需要在系统上安装如下的包：
```
sudo apt-get install libssl-dev libcurl4-openssl-dev libyaml-dev build-essential libopenblas-dev libcap-dev ffmpeg
```

克隆该仓库（如果您访问Github较慢，国内用户可以使用[Gitee镜像](https://gitee.com/daochenzha/tods)）:
```
git clone https://github.com/datamllab/tods.git
```
用`pip`在本地安装:
```
cd tods
pip install -e .
```

# 举例
例子在 [/examples](examples/) 中. 对于最基本的使用，你可以评估某个管道在某数据集上的表现。下面我们提供的例子演示了如何加载默认的管道，并评估它在yahoo数据集的子集上的表现。
```python
import pandas as pd

from tods import schemas as schemas_utils
from tods import generate_dataset, evaluate_pipeline

table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
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
我们也提供AutoML的支持来自动帮您找到最适合您数据的管道。
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
# 致谢
我们诚挚地感谢DRAPA的Data Driven Discovery of Models (D3M)项目。

