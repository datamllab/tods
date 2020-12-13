import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodCOF import COFPrimitive

class COFSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=COFPrimitive, **hyperparams)

