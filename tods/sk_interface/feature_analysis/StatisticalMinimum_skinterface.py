import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalMinimum import StatisticalMinimumPrimitive

class StatisticalMinimumSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalMinimumPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
