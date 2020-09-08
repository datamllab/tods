from d3m.primitive_interfaces.base import *

__all__ = ('SupervisedLearnerPrimitiveBase',)


class SupervisedLearnerPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which have to be fitted on both input and output data
    before they can start producing (useful) outputs from inputs.
    """
