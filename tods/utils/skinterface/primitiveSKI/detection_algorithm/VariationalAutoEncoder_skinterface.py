import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodVAE import VariationalAutoEncoderPrimitive

class VariationalAutoEncoderSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=VariationalAutoEncoderPrimitive, **hyperparams)

