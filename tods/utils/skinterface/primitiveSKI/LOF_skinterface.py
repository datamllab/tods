import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodLOF import LOFPrimitive

class LOFSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LOFPrimitive, **hyperparams)

