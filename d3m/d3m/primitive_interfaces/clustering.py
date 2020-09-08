import abc
import typing

from d3m import types, utils
from d3m.primitive_interfaces.base import *
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

__all__ = ('ClusteringLearnerPrimitiveBase', 'ClusteringTransformerPrimitiveBase', 'DistanceMatrixOutput', 'ClusteringDistanceMatrixMixin')

DistanceMatrixOutput = typing.TypeVar('DistanceMatrixOutput', bound=typing.Union[types.Container])  # type: ignore


class ClusteringLearnerPrimitiveBase(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives implementing a clustering algorithm which learns clusters.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        ``produce`` method should return a membership map.

        A data structure that for each input sample tells to which cluster that sample was assigned to. So ``Outputs``
        should have the same number of samples than ``Inputs``, and the value at each output sample should represent
        a cluster. Consider representing it with just a simple numeric identifier.

        Parameters
        ----------
        inputs:
            The inputs of shape [num_inputs, ...].
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        The outputs of shape [num_inputs, 1] wrapped inside ``CallResult`` for a simple numeric
        cluster identifier.
        """


class ClusteringTransformerPrimitiveBase(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives implementing a clustering algorithm without learning any sort of model.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        ``produce`` method should return a membership map.

        A data structure that for each input sample tells to which cluster that sample was assigned to. So ``Outputs``
        should have the same number of samples than ``Inputs``, and the value at each output sample should represent
        a cluster. Consider representing it with just a simple numeric identifier.

        If an implementation of this method computes clusters based on the whole set of input samples,
        use ``inputs_across_samples`` decorator to mark ``inputs`` as being computed across samples.

        Parameters
        ----------
        inputs:
            The inputs of shape [num_inputs, ...].
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        The outputs of shape [num_inputs, 1] wrapped inside ``CallResult`` for a simple numeric
        cluster identifier.
        """


class ClusteringDistanceMatrixMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams, DistanceMatrixOutput], metaclass=utils.GenericMetaclass):
    @abc.abstractmethod
    def produce_distance_matrix(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[DistanceMatrixOutput]:
        """
        Semantics of this call are the same as the call to a regular ``produce`` method, just
        that the output is a distance matrix instead of a membership map.

        Implementations of this method should use ``inputs_across_samples`` decorator to mark ``inputs``
        as being computed across samples.

        When this mixin is used with `ClusteringTransformerPrimitiveBase`, ``Params`` type variable should
        be set to ``None``.

        Parameters
        ----------
        inputs:
            The inputs of shape [num_inputs, ...].
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        The distance matrix of shape [num_inputs, num_inputs, ...] wrapped inside ``CallResult``, where (i, j) element
        of the matrix represent a distance between i-th and j-th sample in the inputs.
        """
