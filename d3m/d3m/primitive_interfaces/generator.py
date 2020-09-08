import abc
import typing

from d3m import container
from d3m.primitive_interfaces.base import *

__all__ = ('GeneratorPrimitiveBase',)


class GeneratorPrimitiveBase(PrimitiveBase[container.List, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which have to be fitted before they can start
    producing (useful) outputs, but they are fitted only on output data.
    Moreover, they do not accept any inputs to generate outputs,
    which is represented as a sequence (list) of non-negative integer values
    to ``produce`` method, only to signal how many outputs are requested, and
    which one from the potential set of outputs.

    The list of integer values to ``produce`` method provides support for batching.
    A caller does not have to rely on the order in which the primitive is called
    but can specify the index of the requested output.

    This class is parameterized using only by three type variables,
    ``Outputs``, ``Params``, and ``Hyperparams``.
    """

    @abc.abstractmethod
    def set_training_data(self, *, outputs: Outputs) -> None:  # type: ignore
        """
        Sets training data of this primitive.

        Parameters
        ----------
        outputs:
            The outputs.
        """

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: container.List, outputs: Outputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The inputs given to all produce methods.
        outputs:
            The outputs given to ``set_training_data``.
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

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, outputs=outputs)  # type: ignore
