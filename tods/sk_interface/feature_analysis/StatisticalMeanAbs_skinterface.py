import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalMeanAbs import StatisticalMeanAbsPrimitive

class StatisticalMeanAbsSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalMeanAbsPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
