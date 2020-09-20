from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout , LSTM
from keras.regularizers import l2
from keras.losses import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.models.base import BaseDetector

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
# from numba import njit
from pyod.utils.utility import argmaxn

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas
import uuid

from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase



__all__ = ('DeepLog',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    hidden_size = hyperparams.Hyperparameter[int](
        default=64,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="hidden state dimension"
    )

    loss = hyperparams.Hyperparameter[typing.Union[str, None]](
        default='mean_squared_error',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="loss function"
    )

    optimizer = hyperparams.Hyperparameter[typing.Union[str, None]](
        default='Adam',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Optimizer"
    )

    epochs = hyperparams.Hyperparameter[int](
        default=10,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Epoch"
    )

    batch_size = hyperparams.Hyperparameter[int](
        default=32,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Batch size"
    )

    dropout_rate = hyperparams.Hyperparameter[float](
        default=0.2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Dropout rate"
    )

    l2_regularizer = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="l2 regularizer"
    )

    validation_size = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="validation size"
    )

    window_size = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="window size"
    )

    features = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Number of features in Input"
    )

    stacked_layers = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Number of LSTM layers between input layer and Final Dense Layer"
    )

    preprocessing = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Whether to Preprosses the data"
    )

    verbose = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="verbose"
    )


    contamination = hyperparams.Uniform(
        lower=0.,
        upper=0.5,
        default=0.1,
        description='the amount of contamination of the data set, i.e.the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class DeepLogPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that uses DeepLog for outlier detection

    Parameters
        ----------


    """

    __author__ = "DATA Lab at Texas A&M University",
    metadata = metadata_base.PrimitiveMetadata(
        {
        '__author__': "DATA Lab @Texas A&M University",
        'name': "DeepLog Anomolay Detection",
        'python_path': 'd3m.primitives.tods.detection_algorithm.deeplog',
        'source': {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods/-/blob/Yile/anomaly-primitives/anomaly_primitives/MatrixProfile.py']},
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.DEEPLOG], 
        'primitive_family': metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'DeepLogPrimitive')),
        'hyperparams_to_tune': ['hidden_size', 'loss', 'optimizer', 'epochs', 'batch_size',
                                'l2_regularizer', 'validation_size', 
                                'window_size', 'features', 'stacked_layers', 'preprocessing', 'verbose', 'dropout_rate','contamination'],
        'version': '0.0.1', 
        }
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,  #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._clf = DeeplogLstm(hidden_size=hyperparams['hidden_size'],
                                loss=hyperparams['loss'],
                                optimizer=hyperparams['optimizer'],
                                epochs=hyperparams['epochs'],
                                batch_size=hyperparams['batch_size'],
                                dropout_rate=hyperparams['dropout_rate'],
                                l2_regularizer=hyperparams['l2_regularizer'],
                                validation_size=hyperparams['validation_size'],
                                window_size=hyperparams['window_size'],
                                stacked_layers=hyperparams['stacked_layers'],
                                preprocessing=hyperparams['preprocessing'],
                                verbose=hyperparams['verbose'],
                                contamination=hyperparams['contamination']

                                )

    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        super().set_training_data(inputs=inputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        return super().fit()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            1 marks Outliers, 0 marks normal.
        """
        return super().produce(inputs=inputs, timeout=timeout, iterations=iterations)

    def get_params(self) -> Params:
        """
        Return parameters.
        Args:
            None

        Returns:
            class Params
        """
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters for outlier detection.
        Args:
            params: class Params

        Returns:
            None
        """
        super().set_params(params=params)


class DeeplogLstm(BaseDetector):
    """Class to Implement Deep Log LSTM based on "https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
       Only Parameter Value anomaly detection layer has been implemented for time series data"""

    def __init__(self, hidden_size : int = 64,
                 optimizer : str ='adam',loss=mean_squared_error,preprocessing=True,
                 epochs : int =100, batch_size : int =32, dropout_rate : float =0.0,
                 l2_regularizer : float =0.1, validation_size : float =0.1,
                 window_size: int = 1, stacked_layers: int  = 1, verbose : int = 1, contamination:int = 0.001):

        super(DeeplogLstm, self).__init__(contamination=contamination)
        self.hidden_size = hidden_size
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.window_size = window_size
        self.stacked_layers = stacked_layers
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.contamination = contamination





    def _build_model(self):
        """
        Builds Stacked LSTM model.
        Args:
            inputs : Self object containing model parameters

        Returns:
            return : model
        """

        model = Sequential()

        #InputLayer
        model.add(LSTM(self.hidden_size,input_shape = (self.window_size,self.n_features_),return_sequences=True,dropout = self.dropout_rate))
        #stacked layer
        for layers in range(self.stacked_layers):
            if(layers == self.stacked_layers -1 ):
                model.add(LSTM(self.hidden_size, return_sequences=False,dropout = self.dropout_rate))
                continue
            model.add(LSTM(self.hidden_size,return_sequences=True,dropout = self.dropout_rate))
        #output layer

        model.add(Dense(self.n_features_))
        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        if self.verbose >= 1:
            print(model.summary())
        return model

    def fit(self,X,y=None):
        """
        Fit data to  LSTM model.
        Args:
            inputs : X , ndarray of size (number of sample,features)

        Returns:
            return : self object with trained model
        """




        X = check_array(X)
        self._set_n_classes(y)
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]


        X_train,Y_train = self._preprocess_data_for_LSTM(X)

        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_train, Y_train,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        pred_scores = np.zeros(X.shape)
        pred_scores[self.window_size:] = self.model_.predict(X_train)

        Y_train_for_decision_scores = np.zeros(X.shape)
        Y_train_for_decision_scores[self.window_size:] = Y_train
        self.decision_scores_ = pairwise_distances_no_broadcast(Y_train_for_decision_scores,
                                                                pred_scores)

        self._process_decision_scores()
        return self


    def _preprocess_data_for_LSTM(self,X):
        """
        Preposses data and prepare sequence of data based on number of samples needed in a window
        Args:
            inputs : X , ndarray of size (number of sample,features)

        Returns:
            return : X , Y  X being samples till (t-1) of data and Y the t time data
        """
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        X_data = []
        Y_data = []
        for index in range(X.shape[0] - self.window_size):
            X_data.append(X_norm[index:index+self.window_size])
            Y_data.append(X_norm[index+self.window_size])
        X_data = np.asarray(X_data)
        Y_data = np.asarray(Y_data)

        return X_data,Y_data



    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. .
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])

        X = check_array(X)
        print("inside")
        print(X.shape)
        print(X[0])
        X_norm,Y_norm = self._preprocess_data_for_LSTM(X)
        pred_scores = np.zeros(X.shape)
        pred_scores[self.window_size:] = self.model_.predict(X_norm)
        Y_norm_for_decision_scores = np.zeros(X.shape)
        Y_norm_for_decision_scores[self.window_size:] = Y_norm
        return pairwise_distances_no_broadcast(Y_norm_for_decision_scores, pred_scores)






