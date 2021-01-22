from d3m import container
from tods.detection_algorithm import DeepLog
from tods.detection_algorithm.PyodABOD import ABODPrimitive
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive
from tods.detection_algorithm.PyodSOD import SODPrimitive
from tods.detection_algorithm.AutoRegODetect import AutoRegODetectorPrimitive

class BaseSKI():
    def __init__(self, primitive, **hyperparameter):

        hyperparam_buf = list(primitive.metadata.get_hyperparams().defaults().keys())
        hyperparam_input = list(hyperparameter.keys())
        if not set(hyperparam_buf) > set(hyperparam_input):
            invalid_hyperparam = list(set(hyperparam_input) - set(hyperparam_buf))
            raise TypeError(self.__class__.__name__ + ' got unexpected keyword argument ' + str(invalid_hyperparam))

        hyperparams_class = primitive.metadata.get_hyperparams()
        hyperparams = hyperparams_class.defaults()
        #print("items ", type(hyperparameter.items()))
        if len(hyperparameter.items())!=0:
            #for key, value in hyperparameter.items():
            hyperparams = hyperparams.replace(hyperparameter)
            
        self.primitive = primitive(hyperparams=hyperparams)
        self.fit_available = False
        self.predict_available = False
        self.produce_available = False

        # print(hyperparams)

    def transform(self, X):     #transform the ndarray to d3m dataframe, select columns to use
        column_name = [str(col_index) for col_index in range(X.shape[1])]
        return container.DataFrame(X, columns=column_name, generate_metadata=True)

        # use_columns = [iter for iter in range(len(X))]
        # inputs = {}
        # for i in use_columns:
        #   inputs['col_'+str(i)] = list(X[i])
        # inputs = container.DataFrame(inputs, columns=list(inputs.keys()), generate_metadata=True)
        # return inputs

    def set_training_data(self, data):
        return self.primitive.set_training_data(inputs=data)

    def fit(self, data):
        # print(data)

        if not self.fit_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'fit\'')

        data = self.transform(data)
        # print(data)
        self.set_training_data(data)
        
        return self.primitive.fit()
    
    def predict(self, data):

        if not self.predict_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'predict\'')

        data = self.transform(data)
        return self.primitive.produce(inputs=data).value.values
    
    def predict_score(self, data):

        if not self.predict_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'predict_score\'')

        data = self.transform(data)
        return self.primitive.produce_score(inputs=data).value.values

    def produce(self, data):    #produce function for other primitive types

        if not self.produce_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'produce\'')

        data = self.transform(data)
        return self.primitive.produce(inputs=data).value.values
"""
if __name__ == '__main__':
    import numpy as np
    X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])
    X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])
    transformer = SKInterface(AutoRegODetectorPrimitive, contamination=0.2, window_size=2)
    transformer.fit(X_train)
    prediction_labels = transformer.produce(X_test)
    prediction_score = transformer.produce_score(X_test)
    print("Prediction Labels\n", prediction_labels)
    print("Prediction Score\n", prediction_score)
"""

"""
  def transform(self, X):
    inputs = {}
    for i in range(len(X)):
      inputs['col_'+str(i)] = list(X[i])
    inputs = container.DataFrame(inputs, columns=list(inputs.keys()), generate_metadata=True)
    outputs = self.primitive.produce(inputs=inputs).value.to_numpy()
    return outputs

    'contamination': contamination,
          'use_columns': use_columns,
          'return_result': return_result,
"""
#use_columns=(-1,), contamination=0.1, return_result='append'
