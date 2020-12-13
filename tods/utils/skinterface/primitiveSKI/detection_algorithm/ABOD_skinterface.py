import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodABOD import ABODPrimitive

class ABODSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=ABODPrimitive, **hyperparams)

