import numpy as np 
from ..base import BaseSKI
from tods.feature_analysis.SKTruncatedSVD import SKTruncatedSVDPrimitive

class SKTruncatedSVDSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SKTruncatedSVDPrimitive, **hyperparams)
		self.fit_available = True
		self.predict_available = False
		self.produce_available = True
