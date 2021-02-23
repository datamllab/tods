import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalGmean import StatisticalGmeanPrimitive

class StatisticalGmeanSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalGmeanPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
