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
# from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
# from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

from .SODBasePrimitive import Params_SODBase, Hyperparams_SODBase, SupervisedOutlierDetectorBase
from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
# from pyod.models.knn import KNN
from .core.SOD_algorithm_template import SODetector
from typing import Union
import uuid

__all__ = ('SODPrimitive',)

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_SODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_SODBase):
    ######## Add more Hyperparamters #######

    pass


class SODPrimitive(SupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
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
        "name": "TODS.anomaly_detection_primitives.SODPrimitive",
        "python_path": "d3m.primitives.tods.detection_algorithm.sod_primitive",
        "source": {
            'name': "DATA Lab @Taxes A&M University",
            'contact': 'mailto:khlai037@tamu.edu',
        },
        "hyperparams_to_tune": ['use_columns'],
        "version": "0.0.1",
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE,
        ],
        "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'SODPrimitive'))
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = SODetector()

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


