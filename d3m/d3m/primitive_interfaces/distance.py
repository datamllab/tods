import abc
import typing

from d3m import types
from d3m.primitive_interfaces.base import *
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

__all__ = ('PairwiseDistanceLearnerPrimitiveBase', 'PairwiseDistanceTransformerPrimitiveBase', 'InputLabels')

InputLabels = typing.TypeVar('InputLabels', bound=typing.Union[types.Container])  # type: ignore


# Defining Generic with all type variables allows us to specify the order and an additional type variable.
class PairwiseDistanceLearnerPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams], typing.Generic[Inputs, InputLabels, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which learn distances (however defined) between two
    different sets of instances.

    Class is parameterized using five type variables, ``Inputs``, ``InputLabels``, ``Outputs``, ``Params``, and ``Hyperparams``.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:  # type: ignore
        """
        Computes distance matrix between two sets of data.

        Implementations of this method should use ``inputs_across_samples`` decorator to mark ``inputs``
        and ``second_inputs`` as being computed across samples.

        Parameters
        ----------
        inputs:
            The first set of collections of instances.
        second_inputs:
            The second set of collections of instances.
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        ---------
        A n by m distance matrix describing the relationship between each instance in inputs[0] and each instance
        in inputs[1] (n and m are the number of instances in inputs[0] and inputs[1], respectively),
        wrapped inside ``CallResult``.
        """

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs, input_labels: InputLabels) -> None:  # type: ignore
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs:
            The inputs.
        input_labels:
            A set of class labels for the inputs.
        """

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:  # type: ignore
        """
        A method calling multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The first set of collections of instances.
        second_inputs:
            The second set of collections of instances.
        timeout:
            A maximum time this primitive should take to produce outputs for all produce methods
            listed in ``produce_methods`` argument, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, second_inputs=second_inputs)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, input_labels: InputLabels,
                          second_inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:  # type: ignore
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The first set of collections of instances.
        input_labels:
            A set of class labels for the inputs.
        second_inputs:
            The second set of collections of instances.
        timeout:
            A maximum time this primitive should take to both fit the primitive and produce outputs
            for all produce methods listed in ``produce_methods`` argument, in seconds.
        iterations:
            How many of internal iterations should the primitive do for both fitting and producing
            outputs of all produce methods.

        Returns
        -------
        A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, input_labels=input_labels, second_inputs=second_inputs)


class PairwiseDistanceTransformerPrimitiveBase(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives which compute distances (however defined) between two
    different sets of instances without learning any sort of model.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:  # type: ignore
        """
        Computes distance matrix between two sets of data.

        Implementations of this method should use ``inputs_across_samples`` decorator to mark ``inputs``
        and ``second_inputs`` as being computed across samples.

        Parameters
        ----------
        inputs:
            The first set of collections of instances.
        second_inputs:
            The second set of collections of instances.
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        ---------
        A n by m distance matrix describing the relationship between each instance in inputs[0] and each instance
        in inputs[1] (n and m are the number of instances in inputs[0] and inputs[1], respectively),
        wrapped inside ``CallResult``.
        """

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:  # type: ignore
        """
        A method calling multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The first set of collections of instances.
        second_inputs:
            The second set of collections of instances.
        timeout:
            A maximum time this primitive should take to produce outputs for all produce methods
            listed in ``produce_methods`` argument, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, second_inputs=second_inputs)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:  # type: ignore
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The first set of collections of instances.
        second_inputs:
            The second set of collections of instances.
        timeout:
            A maximum time this primitive should take to both fit the primitive and produce outputs
            for all produce methods listed in ``produce_methods`` argument, in seconds.
        iterations:
            How many of internal iterations should the primitive do for both fitting and producing
            outputs of all produce methods.

        Returns
        -------
        A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, second_inputs=second_inputs)
