import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.SKAxiswiseScaler import SKAxiswiseScalerPrimitive

class SKAxiswiseScalerSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SKAxiswiseScalerPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
