import numpy as np 
from ..base import BaseSKI
from tods.data_processing.DuplicationValidation import DuplicationValidationPrimitive

class DuplicationValidationSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=DuplicationValidationPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
