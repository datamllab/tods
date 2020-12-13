import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodHBOS import HBOSPrimitive

class HBOSSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=HBOSPrimitive, **hyperparams)

