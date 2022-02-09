import numpy as np 
from d3m import container
from tods.detection_algorithm.Ensemble import EnsemblePrimitive
from ..base import get_default_hyperparameter

class EnsembleSKI:   # pragma: no cover
	def __init__(self, **hyperparameter):

		hyperparams = get_default_hyperparameter(EnsemblePrimitive, hyperparameter)
		self.primitive = EnsemblePrimitive(hyperparams=hyperparams)

	def fit(self, data):

		data = self._sys_data_check(data)
		data = self._transform(data)
		self.primitive.set_training_data(inputs=data)
		self.primitive.fit()

		return

	def predict(self, data):

		data = self._sys_data_check(data)
		data = self._transform(data)
		output_data = self.primitive.produce(inputs=data).value.values

		return output_data

	def _sys_data_check(self, data):

		if type(data) is np.ndarray and data.ndim == 2 and data.shape[1] == 2:
			return data
		else:
			raise AttributeError('Input data should be n×2 numpy array.')

	def _transform(self, X):
		column_name = ['timestamp','value','system_id','scores']
		X = np.concatenate((np.zeros((X.shape[0], 2)), X), axis=1)
		return container.DataFrame(X, columns=column_name, generate_metadata=True)

	# def __init__(self, **hyperparams):
	# 	super().__init__(primitive=EnsemblePrimitive, **hyperparams)
	# 	self.system_num = None
