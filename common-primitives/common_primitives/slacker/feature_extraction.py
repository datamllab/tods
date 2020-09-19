from collections import defaultdict, OrderedDict
import numpy as np
from scipy import signal
from scipy.sparse import csr_matrix, hstack
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from .base import AbstractFeatureExtractor

class DenseMixedStrategyImputer(BaseEstimator, TransformerMixin):

    def __init__(self, missing_values=np.nan, strategies=None, add_missing_indicator=True, verbose=False):
        self.missing_values = missing_values
        if strategies is None:
            raise ValueError('Must provide strategy.')
        allowed_strategies = ['mean', 'median', 'most_frequent']
        if any(s not in allowed_strategies for s in strategies):
            raise ValueError('Invalid strategy in list.')
        self.strategies = strategies
        self.add_missing_indicator = add_missing_indicator
        self.verbose = verbose

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        # print('n_features',n_features)
        if len(self.strategies) != n_features:
            raise ValueError('Number of strategies must equal number of features.')
        self.impute_strategies = list(set(self.strategies))
        self.impute_indices = [np.array([i for i, x in enumerate(self.strategies) if x == s]) for s in self.impute_strategies]
        self.impute_valid_indices = []
        self.imputers = [SimpleImputer(missing_values=self.missing_values, strategy=s, verbose=self.verbose) for s in
                         self.impute_strategies]
        for indices, imputer in zip(self.impute_indices, self.imputers):
            imputer.fit(X[:, indices])
            valid_mask = np.logical_not(np.isnan(imputer.statistics_))
            self.impute_valid_indices.append(indices[valid_mask])
        return self

    def transform(self, X):
        n_samples, n_features = X.shape
        if len(self.strategies) != n_features:
            raise ValueError('Number of strategies must equal number of features.')
        check_is_fitted(self, 'imputers')

        if self.add_missing_indicator:
            output_scale = 2
        else:
            output_scale = 1

        X_out = np.zeros((n_samples, output_scale*n_features))
        for input_indices, output_indices, imputer in zip(self.impute_indices, self.impute_valid_indices, self.imputers):
            X_out[:, output_scale*output_indices] = imputer.transform(X[:, input_indices])

        if self.add_missing_indicator:
            X_out[:, np.arange(1, 2*n_features, 2)] = np.isnan(X).astype('float', copy=False)

        return X_out


class DataFrameCategoricalEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.code_maps = {}
        for k in X.columns:
            self.code_maps[k] = defaultdict(lambda: np.nan)
            self.code_maps[k].update({v: k for k, v in enumerate(X[k].astype('category').cat.categories)})
            self.code_maps[k] = dict(self.code_maps[k])
        return self

    def transform(self, X):
        if set(X.columns) != set(self.code_maps):
            raise ValueError('Columns do not match fit model.')
        return X.apply(lambda x: x.apply(lambda y: self.code_maps[x.name][y])).values


class AnnotatedTabularExtractor(AbstractFeatureExtractor):

    param_distributions = {
        'normalize_text': [True, False],
        'categorize': [True, False],
        'numeric_strategy': ['mean', 'median'],
        'add_missing_indicator': [True, False]
    }

    def __init__(self, normalize_text=False, categorize=False, numeric_strategy='mean', add_missing_indicator=True):
        self.normalize_text = normalize_text
        self.categorize = categorize
        self.numeric_strategy = numeric_strategy
        self.add_missing_indicator = add_missing_indicator

    def set_cols_info(self, cols_info):
        self.cols_info = cols_info

    def determine_colType(self, column):
        variables = self.cols_info
        for var in variables:
            var_colName = var['colName']
            if str(var_colName) != str(column):
                continue
            var_colType = var['colType']
            if var_colType in {'categorical', 'boolean'}:
                return 'categorical'
            elif var_colType in {'integer', 'real'}:
                return 'numeric'
            elif var_colType == 'string':
                return 'text'
            elif var_colType == 'dateTime':
                raise RuntimeError('datTime not implemented in this feature extractor yet !!')


    def fit_transform(self, df, variables):
        df = self.copy_normalize_text(df)

        self.column_types = OrderedDict()

        for column in df:
            itype = self.determine_colType(column)
            # print('itype',itype)
            self.column_types[column] = itype

        self.numeric_columns = [column for column, type in self.column_types.items() if type == 'numeric']
        self.categorical_columns = [column for column, type in self.column_types.items() if type == 'categorical']
        self.text_columns = [column for column, type in self.column_types.items() if type == 'text']

        output_arrays = []

        if len(self.numeric_columns) > 0:
            X = df[self.numeric_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).values
            self.numeric_imputer = DenseMixedStrategyImputer(
                strategies=[self.numeric_strategy]*len(self.numeric_columns),
                add_missing_indicator=self.add_missing_indicator
            )
            X = self.numeric_imputer.fit_transform(X)
            self.numeric_scaler = StandardScaler()
            output_arrays.append(self.numeric_scaler.fit_transform(X))

        if len(self.categorical_columns) > 0:
            self.categorical_encoder = DataFrameCategoricalEncoder()
            X = self.categorical_encoder.fit_transform(df[self.categorical_columns])
            self.categorical_imputer = DenseMixedStrategyImputer(
                strategies=['most_frequent']*len(self.categorical_columns),
                add_missing_indicator=self.add_missing_indicator
            )
            X = self.categorical_imputer.fit_transform(X)
            self.one_hot_encoder = OneHotEncoder(categories='auto')
            output_arrays.append(self.one_hot_encoder.fit_transform(X))

        return hstack([csr_matrix(X) for X in output_arrays], format='csr')

    def transform(self, df):

        check_is_fitted(self, 'column_types')
        if list(df) != list(self.column_types):
            raise ValueError('Data to be transformed does not match fitting data.')

        df = self.copy_normalize_text(df)

        output_arrays = []

        if len(self.numeric_columns) > 0:
            X = df[self.numeric_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).values
            output_arrays.append(self.numeric_scaler.transform(self.numeric_imputer.transform(X)))

        if len(self.categorical_columns) > 0:
            X = self.categorical_encoder.transform(df[self.categorical_columns])
            output_arrays.append(self.one_hot_encoder.transform(self.categorical_imputer.transform(X)))

        return hstack([csr_matrix(X) for X in output_arrays], format='csr')

    def copy_normalize_text(self, df):
        df = df.copy()
        if self.normalize_text:
            for column in df:
                try:
                    df[column] = df[column].str.lower().str.strip()
                except:
                    df[column] = df[column]
        return df
