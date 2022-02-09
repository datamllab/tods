import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.SimpleExponentialSmoothing import SimpleExponentialSmoothingPrimitive

class SimpleExponentialSmoothingSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SimpleExponentialSmoothingPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
