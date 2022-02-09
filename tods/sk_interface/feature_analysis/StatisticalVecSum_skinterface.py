import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalVecSum import StatisticalVecSumPrimitive

class StatisticalVecSumSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalVecSumPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
