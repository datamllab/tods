import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.MatrixProfile import MatrixProfilePrimitive

class MatrixProfileSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=MatrixProfilePrimitive, **hyperparams)

