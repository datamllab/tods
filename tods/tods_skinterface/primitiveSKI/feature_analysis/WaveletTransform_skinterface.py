import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.feature_analysis.WaveletTransform import WaveletTransformPrimitive

class WaveletTransformSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=WaveletTransformPrimitive, **hyperparams)
		self.fit_available = False
		self.predict_available = False
		self.produce_available = True
