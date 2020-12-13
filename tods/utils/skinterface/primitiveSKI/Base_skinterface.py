from d3m import container
from tods.detection_algorithm import DeepLog
from tods.detection_algorithm.PyodABOD import ABODPrimitive
from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive
from tods.detection_algorithm.PyodSOD import SODPrimitive
from tods.detection_algorithm.AutoRegODetect import AutoRegODetectorPrimitive

class BaseSKI():
    def __init__(self, primitive, **hyperparameter):
        hyperparams_class = primitive.metadata.get_hyperparams()
        hyperparams = hyperparams_class.defaults()
        #print("items ", type(hyperparameter.items()))
        if len(hyperparameter.items())!=0:
            #for key, value in hyperparameter.items():
            hyperparams = hyperparams.replace(hyperparameter)
            
        self.primitive = primitive(hyperparams=hyperparams)
        self.use_columns = hyperparams['use_columns']
        #print(hyperparams)

    def transform(self, X):     #transform the ndarray to d3m dataframe, select columns to use
        if self.use_columns==():
            self.use_columns = [iter for iter in range(len(X))]         
        else:
            pass

        inputs = {}
        for i in self.use_columns:
          inputs['col_'+str(i)] = list(X[i])
        inputs = container.DataFrame(inputs, columns=list(inputs.keys()), generate_metadata=True)
        return inputs

    def set_training_data(self, data):
        return self.primitive.set_training_data(inputs=data)

    def fit(self, data):
        data = self.transform(data)
        self.set_training_data(data)
        return self.primitive.fit()
    
    def predict(self, data):
        data = self.transform(data)
        return self.primitive.produce(inputs=data).value
    
    def predict_score(self, data):
        data = self.transform(data)
        return self.primitive.produce_score(inputs=data).value

    def produce(self, data):    #produce function for other primitive types
        data = self.transform(data)
        return self.primitive.produce(inputs=data).value
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