import numpy as np 
from ..base import BaseSKI
from tods.detection_algorithm.SystemWiseDetection import SystemWiseDetectionPrimitive

class SystemWiseDetectionSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SystemWiseDetectionPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = True
		self.produce_available = False
