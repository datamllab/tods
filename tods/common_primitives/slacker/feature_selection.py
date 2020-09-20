from unittest import TestCase

from .base import AbstractFeatureSelector

import numpy as np
from scipy import stats
from scipy.sparse import issparse

from sklearn.feature_selection import f_classif, SelectFromModel, SelectPercentile
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.utils import check_X_y
from sklearn.utils.extmath import safe_sparse_dot, row_norms
from scipy.linalg import norm


# modified to address the issue of centering sparse matrices with a bit of algebra
def better_f_regression(X, y, center=True):
    """Univariate linear regression tests.

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 2 steps:

    1. The cross correlation between each regressor and the target is computed,
       that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
       std(y)).
    2. It is converted to an F score then to a p-value.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}  shape = (n_samples, n_features)
        The set of regressors that will be tested sequentially.

    y : array of shape(n_samples).
        The data matrix

    center : True, bool,
        If true, X and y will be centered.

    Returns
    -------
    F : array, shape=(n_features,)
        F values of features.

    pval : array, shape=(n_features,)
        p-values of F-scores.

    See also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    """
    X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64)
    n_samples = X.shape[0]

    if center:
        y = y - np.mean(y)
        if issparse(X):
            X_means = X.mean(axis=0).getA1()
        else:
            X_means = X.mean(axis=0)
        X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples*X_means**2)
    else:
        X_norms = row_norms(X.T)

    # compute the correlation
    corr = safe_sparse_dot(y, X)
    corr /= X_norms
    corr /= norm(y)

    # convert to p-value
    degrees_of_freedom = y.size - (2 if center else 1)
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)
    return F, pv


class SelectFromLinearSVC(AbstractFeatureSelector):

    param_distributions = {
        'threshold': (1e-5,),
        'C': [float(x) for x in np.logspace(-2, 5, 100)]
    }

    def __init__(self, threshold=None, penalty='l1', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, fit_intercept=True, random_state=None, max_iter=1000):
        self.threshold = threshold
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y):
        self.linear_svc = LinearSVC(penalty=self.penalty, loss=self.loss, dual=self.dual, tol=self.tol,
                                    fit_intercept=self.fit_intercept, random_state=self.random_state,
                                    max_iter=self.max_iter)
        self.linear_svc.fit(X, y)
        self.select_from_model = SelectFromModel(self.linear_svc, threshold=self.threshold, prefit=True)
        return self

    def _get_support_mask(self):
        return self.select_from_model._get_support_mask()

class SelectPercentileClassification(AbstractFeatureSelector, SelectPercentile):

    param_distributions = {
        'score_func': ('f_classif',),
        'percentile': [int(x) for x in np.linspace(10, 100, 100)]
    }

    score_funcs = {
        'f_classif': f_classif
    }

    def __init__(self, *args, **kwargs):
        if 'score_func' in kwargs:
            kwargs['score_func'] = self.score_funcs[kwargs['score_func']]
        super().__init__(*args, **kwargs)


class SelectFromLasso(AbstractFeatureSelector):

    param_distributions = {
        'threshold': (1e-5,),
        'alpha': [float(x) for x in np.logspace(-5, 2, 100)]
    }

    def __init__(self, threshold=None, alpha=1.0, fit_intercept=True, normalize=False, max_iter=1000, tol=0.0001, positive=False, selection='cyclic', random_state=None):
        self.threshold = threshold
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.selection = selection
        self.random_state = random_state

    def fit(self, X, y):
        # NOTE: y is an ndarray of strings
        self.lasso = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, normalize=self.normalize,
                           max_iter=self.max_iter, tol=self.tol, positive=self.positive, selection=self.selection,
                           random_state=self.random_state)
        self.lasso.fit(X, y)
        self.select_from_model = SelectFromModel(self.lasso, threshold=self.threshold, prefit=True)
        return self

    def _get_support_mask(self):
        return self.select_from_model._get_support_mask()


class SelectPercentileRegression(AbstractFeatureSelector, SelectPercentile):

    param_distributions = {
        'score_func': ('f_regression',),
        'percentile': [int(x) for x in np.linspace(10, 100, 100)]
    }

    score_funcs = {
        'f_regression': better_f_regression
    }

    def __init__(self, *args, **kwargs):
        if 'score_func' in kwargs:
            kwargs['score_func'] = self.score_funcs[kwargs['score_func']]
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        # NOTE: y is an ndarray of strings
        super().fit(X, y)
        return self

