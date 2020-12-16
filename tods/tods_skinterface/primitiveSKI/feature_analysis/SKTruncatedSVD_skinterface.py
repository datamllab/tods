import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.SKTruncatedSVD import SKTruncatedSVDPrimitive

class SKTruncatedSVDSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=SKTruncatedSVDPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
