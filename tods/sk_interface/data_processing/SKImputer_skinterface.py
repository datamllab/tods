import numpy as np 
from ..base import BaseSKI
from tods.data_processing.SKImputer import SKImputerPrimitive

class SKImputerSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SKImputerPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
