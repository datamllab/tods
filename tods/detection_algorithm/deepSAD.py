from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

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

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

from .SODBasePrimitive import Params_SODBase, Hyperparams_SODBase, SupervisedOutlierDetectorBase
from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from typing import Union
import uuid

__all__ = ('deepSADPrimitive',)

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_SODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_SODBase):
    ######## Add more Hyperparamters #######


    epochs = hyperparams.Hyperparameter[int](
        default=10,
        description='Number of epochs to train the model.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    dropout_rate = hyperparams.Uniform(  # Hyperparameter[float](
        lower=0.,
        upper=1.,
        default=0.1,
        description='The dropout to be used across all layers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    feature_dim = hyperparams.Hyperparameter[int](
        default=1,
        description='Feature dim of time series data.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    hidden_dim = hyperparams.Hyperparameter[int](
        default=8,
        description='Hidden dim of MLP.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    activation = hyperparams.Union[Union[str, None]](
        configuration=OrderedDict(
            init=hyperparams.Enumeration[str](
                values=['relu', 'sigmoid', 'selu', 'tanh', 'softplus', 'softsign'],
                default='relu',
                description='Method to vote relative_error in a collect attack.',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='Activations function of LSTMs input and hidden layers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class deepSADPrimitive(SupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Template of the wrapper of Supervised Oulier Detector
    Parameters
    ----------
    Add the parameters here.
    Attributes
    ----------
    Add the attributes here.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "__author__": "DATA Lab at Texas A&M University",
        "name": "deepSAD anomaly detection",
        "python_path": "d3m.primitives.tods.detection_algorithm.deepSAD",
        "source": {
            'name': "DATA Lab @Taxes A&M University",
            'contact': 'mailto:khlai037@tamu.edu',
        },
        "hyperparams_to_tune": [ 'epochs', 'dropout_rate', 'feature_dim', 'hidden_dim', 'activation'],
        "version": "0.0.1",
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE,
        ],
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'deepSADPrimitive'))
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = deepSAD( epochs=hyperparams['epochs'],
                                          batch_size=hyperparams['batch_size'],
                                          hidden_dim=hyperparams['hidden_dim'],
                                          activation=hyperparams['activation'],
                                          feature_dim=hyperparams['feature_dim'],
                                          )

        return

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame
            outputs: Container DataFrame of label
        Returns:
            None
        """
        super().set_training_data(inputs=inputs, outputs=outputs)

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



class deepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0, lr: float = 0.001, epochs: int = 50,
               batch_size: int = 128, weight_decay: float = 1e-6,feature_dim=9,
                    hidden_dim=1, activation: str = 'relu'):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c
        self.eps = 1e-6

        self.model=None
        #self.net = None   neural network phi

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.eps = 1e-6
        self.lr = lr
        self.epochs = epochs
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.activation = activation
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)


        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }


    def set_network(self):
        """Builds the neural network phi.---mlp"""

        model = Sequential()
        model.add(Dense(self.hidden_dim, input_dim=self.feature_dim, activation=self.activation))
        model.add(Dense(units=1))
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def train(self,x_train,y_train,lr=0.001,epochs=15):
        """Trains the Deep SAD model on the training data."""

        #self.train_dataset = train_dataset
        self.lr = lr
        self.epochs = epochs
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr,beta_1=self.beta1, beta_2= self.beta2,epsilon=1e-08 )

        self.model= self.set_network()

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            self.init_center_c(self.model,x_train)

        # Training
        # self.net.train()
        for epoch in np.arange(self.epochs):
            for inputs, semi_labels in zip(x_train, y_train):
                loss = self.train_step(inputs,semi_labels)

        return


    def train_step(self,inputs,semi_labels):
        with tf.GradientTape() as tape:
            outputs = self.model(inputs,training=True)
            dist = tf.reduce_sum((outputs-self.c)**2,1)
            losses = tf.where(semi_labels == 0,dist,self.eta*((dist + self.eps)**tf.cast(semi_labels,dtype = tf.float32)))
            loss = tf.reduce_mean(losses)

        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))

        return loss


    def test(self, x_test, y_test):
        """Tests the Deep SAD model on the test data."""

        # Get test data loader
        #_, test_loader = dataset.loaders(batch_size=self.batch_size)
        print('Starting Test')
        input_list = []
        label_list = []
        score_list = []

        # Testing
        # net.eval()
        for inputs,labels in x_test, y_test:
            outputs= self.model(inputs,training=False)
            dist = tf.reduce_sum((outputs-self.c)**2,1)
            input_list.append((inputs))
            label_list.append(labels)
            score_list.append(dist)

        input = tf.concat(input_list,axis=0).numpy()
        labels = tf.concat(label_list,axis=0).numpy()
        scores = tf.concat(score_list,axis=0).numpy()

        return scores, input,labels



    def init_center_c(self, xtrain, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        print('Initialize center c')
        n_samples = 0
        c = tf.zeros(self.model.rep_dim)

        for inputs,labels,semi in xtrain:
            # get the inputs of the batch
            outputs = self.model(inputs)
            n_samples += outputs.shape[0]
            c += tf.reduce_sum(outputs,0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c = tf.where((c >= 0) & (c < eps), eps, c)
        c = tf.where((c < 0) & (c > -eps), -eps, c)

        self.c = c

