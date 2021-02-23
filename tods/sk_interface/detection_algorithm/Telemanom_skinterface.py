import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.Telemanom import TelemanomPrimitive

class TelemanomSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=TelemanomPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
