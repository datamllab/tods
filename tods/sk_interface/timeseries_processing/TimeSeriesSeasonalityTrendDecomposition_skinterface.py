import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.TimeSeriesSeasonalityTrendDecomposition import TimeSeriesSeasonalityTrendDecompositionPrimitive

class TimeSeriesSeasonalityTrendDecompositionSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=TimeSeriesSeasonalityTrendDecompositionPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
