import numpy as np 
from ..Base_skinterface import BaseSKI
from tods.detection_algorithm.LSTMODetect import LSTMODetectorPrimitive

class LSTMODetectorSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=LSTMODetectorPrimitive, **hyperparams)

