import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.PyodSoGaal import So_GaalPrimitive

class So_GaalSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=So_GaalPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
