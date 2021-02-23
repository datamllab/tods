import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive

class AutoEncoderSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=AutoEncoderPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
