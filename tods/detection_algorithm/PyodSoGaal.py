from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
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

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from pyod.models.so_gaal import SO_GAAL
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    ######## Add more Hyperparamters #######




    stop_epochs = hyperparams.Hyperparameter[int](
        default=20,
        description='Number of epochs to train the model.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    lr_d = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.01,
        description='The learn rate of the discriminator. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    lr_g = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.0001,
        description='The learn rate of the generator.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    decay = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=1e-6,
        description='The decay parameter for SGD',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    momentum = hyperparams.Uniform(
        lower=0.,
        upper=1.,
        default=0.9,
        description='The momentum parameter for SGD',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    contamination = hyperparams.Uniform(
        lower=0.,
        upper=0.5,
        default=0.1,
        description='the amount of contamination of the data set, i.e.the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function',
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




class So_GaalPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """Single-Objective Generative Adversarial Active Learning.
    SO-GAAL directly generates informative potential outliers to assist the
    classifier in describing a boundary that can separate outliers from normal
    data effectively. Moreover, to prevent the generator from falling into the
    mode collapsing problem, the network structure of SO-GAAL is expanded from
    a single generator (SO-GAAL) to multiple generators with different
    objectives (MO-GAAL) to generate a reasonable reference distribution for
    the whole dataset.
    Read more in the :cite:`liu2019generative`.
    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.
    stop_epochs : int, optional (default=20)
        The number of epochs of training.
    lr_d : float, optional (default=0.01)
        The learn rate of the discriminator.
    lr_g : float, optional (default=0.0001)
        The learn rate of the generator.
    decay : float, optional (default=1e-6)
        The decay parameter for SGD.
    momentum : float, optional (default=0.9)
        The momentum parameter for SGD.
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.
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

    __author__ = "DATA Lab at Texas A&M University",
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '56e6cfe9-d9e9-495f-83da-cfed6fa27da1',
            'version': '0.1.0',
            'name': 'So_Gaal Anomaly Detection',
            'python_path': 'd3m.primitives.tods.detection_algorithm.pyod_sogaal',
            'keywords': ['Time Series', 'GAN'],
            "hyperparams_to_tune": ['stop_epochs','lr_d','lr_g','decay','momentum'],
            'source': {
                'name': 'DATA Lab at Texas A&M University',
                'uris': ['https://gitlab.com/lhenry15/tods.git',
                         'https://gitlab.com/lhenry15/tods/-/blob/devesh/tods/detection_algorithm/PyodSoGaal.py'],
                'contact': 'mailto:khlai037@tamu.edu'

            },
            'installation': [
                {'type': metadata_base.PrimitiveInstallationType.PIP,
                 'package_uri': 'git+https://gitlab.com/lhenry15/tods.git@{git_commit}#egg=TODS'.format(
                     git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                 ),
                 }

            ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_PROFILING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,

        }
    )

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = SO_GAAL(stop_epochs=hyperparams['stop_epochs'],
                        lr_d=hyperparams['lr_d'],
                        lr_g=hyperparams['lr_g'],
                        decay=hyperparams['decay'],
                        momentum=hyperparams['momentum'],
                        contamination=hyperparams['contamination'],

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
