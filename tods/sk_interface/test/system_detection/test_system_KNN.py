import numpy as np

from tods.sk_interface.feature_analysis.StatisticalMaximum_skinterface import StatisticalMaximumSKI
from tods.sk_interface.detection_algorithm.KNN_skinterface import KNNSKI
from tods.sk_interface.data_ensemble.Ensemble_skinterface import EnsembleSKI
from tods.sk_interface.utils.data import generate_3D_data, load_sys_data, generate_sys_feature

# Generate 3D data (n, T, d), n: system number, T: time, d: dimension

# n_sys = 5
# X_train, y_train, X_test, y_test = generate_3D_data(n_sys=n_sys,
#                                                     n_train=1000,
#                                                     n_test=1000,
#                                                     n_features=3,
#                                                     contamination=0.1)

X_train, y_train, sys_info_train = load_sys_data('../../../../datasets/anomaly/system_wise/sample/train.csv',
                                 '../../../../datasets/anomaly/system_wise/sample/systems')
X_test, y_test, sys_info_test = load_sys_data('../../../../datasets/anomaly/system_wise/sample/train.csv',
                                 '../../../../datasets/anomaly/system_wise/sample/systems')
n_sys = sys_info_train['sys_num']

# feature analysis algorithms
stmax = StatisticalMaximumSKI(system_num=n_sys)

# OD algorithms
detection_module = KNNSKI(contamination=0.1, system_num=n_sys)

# ensemble model
ensemble_module = EnsembleSKI()

# Fit the feature analysis algorithms
X_train = stmax.produce(X_train)
X_test = stmax.produce(X_test)

# Fit the detector
detection_module.fit(X_train)
sys_ts_score = detection_module.predict_score(X_test) # shape (n, T, 1)

# generate sys_feature based on the time-series anomaly score
sys_feature = generate_sys_feature(sys_ts_score) # shape (T, n)

print(sys_feature.shape)
print(sys_feature.ndim)

# Ensemble the time series outlier socre for each system
ensemble_module.fit(sys_feature)
sys_score = ensemble_module.predict(sys_feature)

print(sys_score)

