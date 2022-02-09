import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalSkew import StatisticalSkewPrimitive

class StatisticalSkewSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalSkewPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
