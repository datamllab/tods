import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodOCSVM import OCSVMPrimitive

class OCSVMSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=OCSVMPrimitive, **hyperparams)

