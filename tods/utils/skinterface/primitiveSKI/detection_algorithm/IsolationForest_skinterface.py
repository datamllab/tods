import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodIsolationForest import IsolationForestPrimitive

class IsolationForestSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=IsolationForestPrimitive, **hyperparams)

