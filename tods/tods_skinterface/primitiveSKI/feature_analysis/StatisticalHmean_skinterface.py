import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.StatisticalHmean import StatisticalHmeanPrimitive

class StatisticalHmeanSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalHmeanPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
