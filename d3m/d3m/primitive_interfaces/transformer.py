import typing

from d3m.primitive_interfaces.base import *

__all__ = ('TransformerPrimitiveBase',)


class TransformerPrimitiveBase(PrimitiveBase[Inputs, Outputs, None, Hyperparams]):
    """
    A base class for primitives which are not fitted at all and can
    simply produce (useful) outputs from inputs directly. As such they
    also do not have any state (params).

    This class is parameterized using only three type variables, ``Inputs``,
    ``Outputs``, and ``Hyperparams``.
    """

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.

        Parameters
        ----------
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        A noop.
        """

        return CallResult(None)

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:  # type: ignore
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The inputs given to all produce methods.
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
