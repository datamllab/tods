import numpy as np 
from .Base_skinterface import BaseSKI
from tods.detection_algorithm.PyodSoGaal import So_GaalPrimitive

class So_GaalSKI(BaseSKI):
	def __init__(self, **hyperparams):
		super().__init__(primitive=So_GaalPrimitive, **hyperparams)

