from d3m.primitive_interfaces.base import *
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

__all__ = ('FeaturizationLearnerPrimitiveBase', 'FeaturizationTransformerPrimitiveBase')


class FeaturizationLearnerPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    Use this version for featurizers that allow for fitting (for domain-adaptation, data-specific deep
    learning, etc.).  Otherwise use `FeaturizationTransformerPrimitiveBase`.
    """


class FeaturizationTransformerPrimitiveBase(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    Use this version for featurizers that do not require or allow any fitting, and simply
    transform data on demand.  Otherwise use `FeaturizationLearnerPrimitiveBase`.
    """
