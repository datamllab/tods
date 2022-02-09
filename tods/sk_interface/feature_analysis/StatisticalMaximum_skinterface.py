import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalMaximum import StatisticalMaximumPrimitive

class StatisticalMaximumSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalMaximumPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
