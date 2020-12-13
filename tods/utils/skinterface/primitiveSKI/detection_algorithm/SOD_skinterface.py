import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodSOD import SODPrimitive

class SODSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SODPrimitive, **hyperparams)

