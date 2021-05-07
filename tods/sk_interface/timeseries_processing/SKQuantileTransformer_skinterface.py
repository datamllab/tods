import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.SKQuantileTransformer import SKQuantileTransformerPrimitive

class SKQuantileTransformerSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SKQuantileTransformerPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
