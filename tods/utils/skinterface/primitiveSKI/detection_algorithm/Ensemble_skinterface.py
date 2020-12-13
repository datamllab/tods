import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.Ensemble import EnsemblePrimitive

class EnsembleSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=EnsemblePrimitive, **hyperparams)

