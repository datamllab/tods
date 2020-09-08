import abc
import typing

from d3m.primitive_interfaces.base import *

__all__ = ('UnsupervisedLearnerPrimitiveBase',)


class UnsupervisedLearnerPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which have to be fitted before they can start
    producing (useful) outputs from inputs, but they are fitted only on input data.
    """

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs:
            The inputs.
        """

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:  # type: ignore
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The inputs given to ``set_training_data`` and all produce methods.
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

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)
