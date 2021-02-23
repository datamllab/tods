import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.StatisticalAbsEnergy import StatisticalAbsEnergyPrimitive

class StatisticalAbsEnergySKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=StatisticalAbsEnergyPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
