import numpy as np 
from ..base import BaseSKI
from tods.data_processing.TimeStampValidation import TimeStampValidationPrimitive

class TimeStampValidationSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=TimeStampValidationPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
