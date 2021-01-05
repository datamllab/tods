import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.StatisticalStd import StatisticalStdPrimitive

class StatisticalStdSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalStdPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
