from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection.base import SelectorMixin

# https://stackoverflow.com/a/3862957
def get_all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in get_all_subclasses(s)]


def sample_param_distributions(param_distributions):

    try:
        return sample_param_distributions_dict(param_distributions)
    except AttributeError:
        i = np.random.randint(len(param_distributions))
        return sample_param_distributions_dict(param_distributions[i])


def sample_param_distributions_dict(param_distributions_dict):

    params = {}
    for k, v in param_distributions_dict.items():
        i = np.random.randint(len(v))
        params[k] = v[i]
    return params


class AbstractParameterized(ABC):

    param_distributions = {}

    @classmethod
    def get_random_parameters(cls):
        return sample_param_distributions(cls.param_distributions)


class AbstractFeatureExtractor(AbstractParameterized, BaseEstimator):

    def fit(self, df, variables):
        self.fit_transform(df, variables)
        return self

    @abstractmethod
    def fit_transform(self, df, variables):
        """ Fits the feature extractor

        :param df:
        :type df: DataFrame
        :param variables:
        :type variables: list[D3MVariable]
        :return:
        :rtype: csr_matrix
        """
        pass

    @abstractmethod
    def transform(self, df):
        """ Transforms the data

        :param df:
        :type df: DataFrame
        :return:
        :rtype: csr_matrix
        """
        pass


class AbstractFeatureSelector(AbstractParameterized, BaseEstimator, SelectorMixin):

    pass


class AbstractEstimator(AbstractParameterized, BaseEstimator):

    @abstractmethod
    def fit(self, X, y):
        """

        :param X:
        :type X: csr_matrix
        :param y:
        :type y: ndarray
        :return:
        :rtype: AbstractEstimator
        """
        return self

    @abstractmethod
    def predict(self, X):
        """

        :param X:
        :type X: csr_matrix
        :return:
        :rtype: ndarray
        """
        pass
