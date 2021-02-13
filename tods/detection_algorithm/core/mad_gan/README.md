# -- Multivariate Anomaly Detection for Time Series Data with GANs -- #

# MAD-GAN

This repository contains code for the paper, _[MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks](https://arxiv.org/pdf/1901.04997.pdf)_, by Dan Li, Dacheng Chen, Jonathan Goh, and See-Kiong Ng.

MAD-GAN is a refined version of GAN-AD at _[Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series](https://arxiv.org/pdf/1809.04758.pdf)_ The code can be found at https://github.com/LiDan456/GAN-AD

(We are still working on this topic, will upload the completed version later...)

## Overview

We used generative adversarial networks (GANs) to do anomaly detection for time series data.
The GAN framework was **R**GAN, whihc was taken from the paper, _[Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633).
Please refer to https://github.com/ratschlab/RGAN for the original code.

## Quickstart

- Python3

- Please unpack the data.7z file in the data folder before run RGAN.py and AD.py

- To train the model:
  
  """ python RGAN.py --settings_file kdd99 """

- To do anomaly detection:

  """ python AD.py --settings_file kdd99_test"""
  
  """ python AD_Invert.py --settings_file kdd99_test"""

## Data

We apply our method on the SWaT and WADI datasets in the paper, however, we didn't upload the data in this repository. Please refer to https://itrust.sutd.edu.sg/ and send request to iTrust is you want to try the data.

In this repository we used kdd cup 1999 dataset as an example (please unpack the data.7z file in the data folder before run RGAN.py and AD.py). You can also down load the original data at http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

