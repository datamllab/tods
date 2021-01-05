import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.StatisticalSkew import StatisticalSkewPrimitive

class StatisticalSkewSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalSkewPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
