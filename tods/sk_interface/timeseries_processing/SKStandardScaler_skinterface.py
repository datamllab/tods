import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.SKStandardScaler import SKStandardScalerPrimitive

class SKStandardScalerSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SKStandardScalerPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
