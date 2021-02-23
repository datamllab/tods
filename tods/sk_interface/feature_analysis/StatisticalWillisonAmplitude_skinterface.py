import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalWillisonAmplitude import StatisticalWillisonAmplitudePrimitive

class StatisticalWillisonAmplitudeSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalWillisonAmplitudePrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
