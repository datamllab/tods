import numpy as np 
from ..base import BaseSKI
from tods.data_processing.CategoricalToBinary import CategoricalToBinaryPrimitive

class CategoricalToBinarySKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=CategoricalToBinaryPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
