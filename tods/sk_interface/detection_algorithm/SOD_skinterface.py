import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.PyodSOD import SODPrimitive

class SODSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SODPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
