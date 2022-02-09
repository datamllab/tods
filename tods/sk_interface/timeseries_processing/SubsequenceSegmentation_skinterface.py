import numpy as np 
from ..base import BaseSKI
from tods.timeseries_processing.SubsequenceSegmentation import SubsequenceSegmentationPrimitive

class SubsequenceSegmentationSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SubsequenceSegmentationPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
