import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.HoltSmoothing import HoltSmoothingPrimitive

class HoltSmoothingSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=HoltSmoothingPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
