import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.MovingAverageTransformer import MovingAverageTransformerPrimitive

class MovingAverageTransformerSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=MovingAverageTransformerPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
