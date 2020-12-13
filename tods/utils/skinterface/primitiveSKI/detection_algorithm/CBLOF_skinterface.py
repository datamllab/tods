import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodCBLOF import CBLOFPrimitive

class CBLOFSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=CBLOFPrimitive, **hyperparams)

