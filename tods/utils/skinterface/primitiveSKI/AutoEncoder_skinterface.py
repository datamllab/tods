import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive

class AutoEncoderSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=AutoEncoderPrimitive, **hyperparams)

