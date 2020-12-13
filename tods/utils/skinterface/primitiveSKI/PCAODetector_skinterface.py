import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.PCAODetect import PCAODetectorPrimitive

class PCAODetectorSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=PCAODetectorPrimitive, **hyperparams)

