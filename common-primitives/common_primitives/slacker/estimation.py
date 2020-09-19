from .base import AbstractEstimator

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, SGDRegressor


class SGDClassifierEstimator(AbstractEstimator):

    param_distributions = {
        'loss': ('hinge', 'log', 'squared_hinge', 'perceptron'),
        'penalty': ('elasticnet',),
        'alpha': [float(x) for x in np.logspace(-9, 0, 10)],
        'l1_ratio': [float(x) for x in np.linspace(0, 1, 11)],
        'fit_intercept': (True, True, True, False)
    }

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.kwargs['n_iter'] = max(5, int(10**6 / n_samples))
        self.sgd_classifier = SGDClassifier(*self.args, **self.kwargs)
        self.sgd_classifier.fit(X, y)

    def predict(self, X):
        return self.sgd_classifier.predict(X)


class SGDRegressorEstimator(AbstractEstimator):

    param_distributions = {
        'loss': ('squared_loss', 'huber'),
        'penalty': ('elasticnet',),
        'alpha': [float(x) for x in np.logspace(-9, 0, 10)],
        'l1_ratio': [float(x) for x in np.linspace(0, 1, 11)],
        'fit_intercept': (True, True, True, False),
        'epsilon': [float(x) for x in np.logspace(-2, 0, 5)],
        'learning_rate': ('optimal', 'invscaling'),
        'eta0': (0.1, 0.01, 0.001),
        'power_t': [float(x) for x in np.linspace(0, 1, 5)]
    }

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.kwargs['n_iter'] = max(5, int(10**6 / n_samples))
        self.sgd_regressor = SGDRegressor(*self.args, **self.kwargs)
        self.sgd_regressor.fit(X, y)

    def predict(self, X):
        return self.sgd_regressor.predict(X)


# TODO: inherit AbstractEstimator, grab param_distributions from cv_setup_map.py in the old slacker,
class RBFSamplerSGDClassifierEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, gamma=1.0, n_components=100, random_state=None, **kwargs):
        kwargs['random_state'] = random_state
        self.rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
        self.sgdclassifier = SGDClassifier(**kwargs)

    def fit(self, X, y):
        X = self.rbf_sampler.fit_transform(X)
        self.sgdclassifier.fit(X, y)
        return self

    def transform(self, X, y=None):
        return np.sqrt(self.rbf_sampler.n_components) / np.sqrt(2.) * self.rbf_sampler.transform(X)

    def predict(self, X):
        return self.sgdclassifier.predict(self.transform(X))

    def decision_function(self, X):
        return self.sgdclassifier.decision_function(self.transform(X))

# TODO: inherit AbstractEstimator, grab param_distributions from cv_setup_map.py in the old slacker,
class RBFSamplerSGDRegressorEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, gamma=1.0, n_components=100, random_state=None, **kwargs):
        kwargs['random_state'] = random_state
        self.rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
        self.sgdregressor = SGDRegressor(**kwargs)

    def fit(self, X, y):
        X = self.rbf_sampler.fit_transform(X)
        self.sgdregressor.fit(X, y)
        return self

    def transform(self, X, y=None):
        return np.sqrt(self.rbf_sampler.n_components) / np.sqrt(2.) * self.rbf_sampler.transform(X)

    def predict(self, X):
        return self.sgdregressor.predict(self.transform(X))

# TODO: Add kernel SVM
# TODO: Add kernel ridge regressor
# TODO: Add random forests / xgboost