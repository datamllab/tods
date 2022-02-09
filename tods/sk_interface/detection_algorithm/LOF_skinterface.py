import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.PyodLOF import LOFPrimitive

class LOFSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LOFPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
