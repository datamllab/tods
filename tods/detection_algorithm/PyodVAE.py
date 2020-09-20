from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import tensorflow
from tensorflow.keras.losses import mean_squared_error
from tensorflow import keras
from tensorflow.keras import losses,layers
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

from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from pyod.models.vae import VAE

import uuid
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Hyperparamters #######

    encoder_neurons = hyperparams.List(
        default=[4, 2, 4],
        elements=hyperparams.Hyperparameter[int](1),
        description='The number of neurons per hidden layers in encoder.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    decoder_neurons = hyperparams.List(
        default=[4, 4, 4],
        elements=hyperparams.Hyperparameter[int](1),
        description='The number of neurons per hidden layers in decoder.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    hidden_activation = hyperparams.Enumeration[str](
        values=['relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                'tanh', 'selu', 'elu', 'exponential'],
        default='relu',
        description='Activation function to use for hidden layers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    output_activation = hyperparams.Enumeration[str](
        values=['relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                'tanh', 'selu', 'elu', 'exponential'],
        default='sigmoid',
        description='Activation function to use for output layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    loss = hyperparams.Enumeration[str](
        values=['mean_squared_error'],
        default='mean_squared_error',
        description='Loss function.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    gamma = hyperparams.Hyperparameter[float](
        default=1.0,
        description='Coefficient of beta VAE regime. Default is regular VAE.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    capacity = hyperparams.Hyperparameter[float](
        default=0.0,
        description='Maximum capacity of a loss bottle neck.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    optimizer = hyperparams.Enumeration[str](
        values=['SGD', 'RMSprop', 'adam', 'Adadelta', 'Adagrad',
                'Adamax', 'Nadam', 'Ftrl'],
        default='adam',
        description='String (name of optimizer) or optimizer instance.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    epochs = hyperparams.Hyperparameter[int](
        default=100,
        description='Number of epochs to train the model.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    batch_size = hyperparams.Hyperparameter[int](
        default=32,
        description='Number of samples per gradient update.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    dropout_rate = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.2,
        description='The dropout to be used across all layers.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    l2_regularizer = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.1,
        description='The regularization strength of activity_regularizer applied on each layer.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    validation_size = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.1,
        description='The percentage of data to be used for validation.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    preprocessing = hyperparams.UniformBool(
        default=True,
        description='If True, apply standardization on the data.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    verbosity = hyperparams.Enumeration[int](
        values=[0, 1, 2],
        default=1,
        description='Verbosity mode.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    random_state = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=0,
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description='the seed used by the random number generator.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    contamination = hyperparams.Uniform(
        lower=0.,
        upper=0.5,
        default=0.01,
        description='The amount of contamination of the data set, i.e. the proportion of outliers in the data set. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


    pass


class VariationalAutoEncoder(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Auto Encoder (AE) is a type of neural networks for learning useful data
    representations unsupervisedly. Similar to PCA, AE could be used to
    detect outlying objects in the data by calculating the reconstruction
    errors. See :cite:`aggarwal2015outlier` Chapter 3 for details.

    Parameters
    ----------
    hidden_neurons : list, optional (default=[4, 2, 4])
        The number of neurons per hidden layers.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    loss : str or obj, optional (default=keras.losses.mean_squared_error)
        String (name of objective function) or objective function.
        See https://keras.io/losses/

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        Verbosity mode.
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.
        For verbosity >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "name": "TODS.anomaly_detection_primitives.VariationalAutoEncoder",
        "python_path": "d3m.primitives.tods.detection_algorithm.pyod_vae",
        "source": {'name': "DATA Lab at Texas A&M University", 'contact': 'mailto:khlai037@tamu.edu','uris': ['https://gitlab.com/lhenry15/tods.git']},
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.VARIATIONAL_AUTO_ENCODER, ],
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "version": "0.0.1",
        "hyperparameters_to_tune": [''],
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'AutoEncoderPrimitive')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        if hyperparams['loss'] == 'mean_squared_error':
            loss = keras.losses.mean_squared_error
        else:
            raise ValueError('VAE only suports mean squered error for now')


        self._clf = VAE(contamination=hyperparams['contamination'],
                        encoder_neurons=hyperparams['encoder_neurons'],
                        decoder_neurons=hyperparams['decoder_neurons'],
                        hidden_activation=hyperparams['hidden_activation'],
                        output_activation=hyperparams['output_activation'],
                        loss=loss,
                        gamma=hyperparams['gamma'],
                        capacity=hyperparams['capacity'],
                        optimizer=hyperparams['optimizer'],
                        epochs=hyperparams['epochs'],
                        batch_size=hyperparams['batch_size'],
                        dropout_rate=hyperparams['dropout_rate'],
                        l2_regularizer=hyperparams['l2_regularizer'],
                        validation_size=hyperparams['validation_size'],
                        preprocessing=hyperparams['preprocessing'],
                        verbosity=hyperparams['verbosity'],
                        random_state=hyperparams['random_state'],
                        )

        return

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


