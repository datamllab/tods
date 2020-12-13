import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.Telemanom import TelemanomPrimitive

class TelemanomSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=TelemanomPrimitive, **hyperparams)

