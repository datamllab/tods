import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.StatisticalMeanAbsTemporalDerivative import StatisticalMeanAbsTemporalDerivativePrimitive

class StatisticalMeanAbsTemporalDerivativeSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalMeanAbsTemporalDerivativePrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
