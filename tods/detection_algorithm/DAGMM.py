from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple



from d3m.metadata import hyperparams, params, base as metadata_base

from d3m.primitive_interfaces.base import CallResult, DockerContainer

from tods.detection_algorithm.core.dagmm.dagmm import DAGMM
import uuid

from d3m import container, utils as d3m_utils

from tods.detection_algorithm.UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase



__all__ = ('DAGMMPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    comp_hiddens = hyperparams.List(
        default=[16,8,1],
        elements=hyperparams.Hyperparameter[int](1),
        description='Sizes of hidden layers of compression network.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    est_hiddens = hyperparams.List(
        default=[8,4],
        elements=hyperparams.Hyperparameter[int](1),
        description='Sizes of hidden layers of estimation network.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    est_dropout_ratio = hyperparams.Hyperparameter[float](
        default=0.25,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Dropout rate of estimation network"
    )

    minibatch_size = hyperparams.Hyperparameter[int](
        default=3,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Mini Batch size"
    )

    epoch_size = hyperparams.Hyperparameter[int](
        default=500,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Epoch"
    )

    rand_seed = hyperparams.Hyperparameter[int](
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="(optional )random seed used when fit() is called"
    )

    learning_rate = hyperparams.Hyperparameter[float](
        default=0.0001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="learning rate"
    )
    lambda1 = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="a parameter of loss function (for energy term)"
    )
    lambda2 = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="a parameter of loss function"
    )

    normalize = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Specify whether input data need to be normalized."
    )

    contamination = hyperparams.Uniform(
        lower=0.,
        upper=0.5,
        default=0.1,
        description='the amount of contamination of the data set, i.e.the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class DAGMMPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Deep Autoencoding Gaussian Mixture Model
    Parameters
        ----------


    """

    __author__ = "DATA Lab at Texas A&M University",
    metadata = metadata_base.PrimitiveMetadata(
        {
        '__author__': "DATA Lab @Texas A&M University",
        'name': "DAGMM",
        'python_path': 'd3m.primitives.tods.detection_algorithm.dagmm',
        'source': {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods/-/blob/Yile/anomaly-primitives/anomaly_primitives/DAGMM.py']},
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.DEEPLOG],
        'primitive_family': metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'DAGMMPrimitive')),
        'hyperparams_to_tune': ['comp_hiddens','est_hiddens','est_dropout_ratio','minibatch_size','epoch_size','rand_seed',
                                'learning_rate','lambda1','lambda2','contamination'],
        'version': '0.0.1',
        }
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,  #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._clf = DAGMM(comp_hiddens= hyperparams['comp_hiddens'],
                          est_hiddens=hyperparams['est_hiddens'],
                          est_dropout_ratio=hyperparams['est_dropout_ratio'],
                          minibatch_size=hyperparams['minibatch_size'],
                          epoch_size=hyperparams['epoch_size'],
                          random_seed=hyperparams['rand_seed'],
                          learning_rate=hyperparams['learning_rate'],
                          lambda2=hyperparams['lambda2'],
                          normalize=hyperparams['normalize'],
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


