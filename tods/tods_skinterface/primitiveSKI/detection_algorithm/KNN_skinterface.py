import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodKNN import KNNPrimitive

class KNNSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=KNNPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
