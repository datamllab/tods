from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
# import numpy
import typing

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
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

from pyod.models.base import BaseDetector

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from .core.LSTMOD import LSTMOutlierDetector

from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted

from pyod.models.base import BaseDetector
import uuid

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Hyperparamters #######

    train_contamination = hyperparams.Uniform(  # Hyperparameter[float](
        lower=0.,
        upper=0.5,
        default=0.0,
        description='Contamination used to calculate relative_error_threshold.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    min_attack_time = hyperparams.Hyperparameter[int](
        default=5,
        description='The minimum amount of recent time steps that is used to define a collective attack.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    danger_coefficient_weight = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.5,
        description='Weight of danger coefficient in decision score.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    loss_func = hyperparams.Enumeration[str](
        values=['mean_squared_error'],
        default='mean_squared_error',
        description='String (name of objective function).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


    optimizer = hyperparams.Enumeration[str](
        values=['adam', 'sgd', 'rmsprop', 'nadam', 'adamax', 'adadelta', 'adagrad'],
        default='adam',
        description='String (name of optimizer).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    epochs = hyperparams.Hyperparameter[int](
        default=10,
        description='Number of epochs to train the model.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    batch_size = hyperparams.Hyperparameter[int](
        default=32,
        description='Number of samples per gradient update.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    dropout_rate = hyperparams.Uniform( # Hyperparameter[float](
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
        default=16,
        description='Hidden dim of LSTM.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    n_hidden_layer = hyperparams.Hyperparameter[int](
        default=0,
        description='Hidden layer number of LSTM.',
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

    diff_group_method = hyperparams.Enumeration[str](
        values=['average', 'max', 'min'],
        default='average',
        description='Method to vote relative_error in a collect attack.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    pass


class LSTMODetector(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """

    Parameters
    ----------
    window_size : int
        The moving window size.

    step_size : int, optional (default=1)
        The displacement for moving window.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "name": "LSTMODetector",
        "python_path": "d3m.primitives.tods.detection_algorithm.LSTMODetector",
        "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
        'uris': ['https://gitlab.com/lhenry15/tods.git', 'https://gitlab.com/lhenry15/tods/-/blob/Junjie/anomaly-primitives/anomaly_primitives/LSTMOD.py']},
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.ISOLATION_FOREST, ], # up to update
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "version": "0.0.1",
        "hyperparams_to_tune": ['contamination', 'train_contamination', 'min_attack_time',
                                'danger_coefficient_weight', 'loss_func', 'optimizer',
                                'epochs', 'batch_size', 'dropout_rate', 'feature_dim', 'hidden_dim',
                                'n_hidden_layer', 'activation', 'diff_group_method'],
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'LSTMODetector')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = LSTMOutlierDetector(contamination=hyperparams['contamination'],
                                        train_contamination=hyperparams['train_contamination'],
                                        min_attack_time=hyperparams['min_attack_time'],
                                        danger_coefficient_weight=hyperparams['danger_coefficient_weight'],
                                        loss=hyperparams['loss_func'],
                                        optimizer=hyperparams['optimizer'],
                                        epochs=hyperparams['epochs'],
                                        batch_size=hyperparams['batch_size'],
                                        dropout_rate=hyperparams['dropout_rate'],
                                        feature_dim=hyperparams['feature_dim'],
                                        hidden_dim=hyperparams['hidden_dim'],
                                        n_hidden_layer=hyperparams['n_hidden_layer'],
                                        activation=hyperparams['activation'],
                                        diff_group_method=hyperparams['diff_group_method'],
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

