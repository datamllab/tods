import numpy as np 
from ..base import BaseSKI
from tods.data_processing.TimeIntervalTransform import TimeIntervalTransformPrimitive

class TimeIntervalTransformSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=TimeIntervalTransformPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
