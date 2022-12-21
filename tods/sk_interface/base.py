from d3m import container
import numpy as np

def get_default_hyperparameter(primitive, hyperparameter):

    # check if input legal hyperparameter
    hyperparam_buf = list(primitive.metadata.get_hyperparams().defaults().keys())
    hyperparam_input = list(hyperparameter.keys())
    if not set(hyperparam_buf) > set(hyperparam_input):
        invalid_hyperparam = list(set(hyperparam_input) - set(hyperparam_buf))
        raise TypeError(primitive.__name__ + ' got unexpected keyword argument ' + str(invalid_hyperparam))

    hyperparams_class = primitive.metadata.get_hyperparams()
    hyperparams = hyperparams_class.defaults()
    # print("items ", type(hyperparameter.items()))
    if len(hyperparameter.items()) != 0:
        # for key, value in hyperparameter.items():
        hyperparams = hyperparams.replace(hyperparameter)

    return hyperparams

class BaseSKI:
    def __init__(self, primitive, system_num=1, **hyperparameter):

        self.fit_available = True if 'fit' in primitive.__dict__ else False
        self.predict_available = True if 'produce' in primitive.__dict__ else False
        self.predict_score_available = True if 'produce_score' in dir(primitive) else False
        self.produce_available = True if 'produce' in primitive.__dict__ else False

        # print(primitive, self.fit_available, self.predict_available, self.predict_score_available, self.produce_available)

        self.system_num = system_num
        hyperparams = get_default_hyperparameter(primitive, hyperparameter)

        if system_num >= 1:
            self.primitives = [primitive(hyperparams=hyperparams) for sys_idx in range(system_num)]
        else:
            raise AttributeError('BaseSKI must have positive system_num.')


        #print(hyperparams)

    def fit(self, data):

        if not self.fit_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'fit\'')

        data = self._sys_data_check(data)

        for sys_idx, primitive in enumerate(self.primitives):
            sys_data = data[sys_idx]
            sys_data = self._transform(sys_data)
            primitive.set_training_data(inputs=sys_data)
            primitive.fit()

        return
    
    def predict(self, data):

        if not self.predict_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'predict\'')

        data = self._sys_data_check(data)
        output_data = self._forward(data, '_produce')

        return output_data
    
    def predict_score(self, data):

        if not self.predict_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'predict_score\'')

        data = self._sys_data_check(data)
        output_data = self._forward(data, '_produce_score')

        return output_data

    def produce(self, data):    #produce function for other primitive types

        if not self.produce_available:
            raise AttributeError('type object ' + self.__class__.__name__ + ' has no attribute \'produce\'')

        data = self._sys_data_check(data)
        output_data = self._forward(data, 'produce')

        return output_data

    def _sys_data_check(self, data):
        if self.system_num == 1:
            if type(data) is np.ndarray and data.ndim == 2:
                data = [data] # np.expand_dims(data, axis=0)
            else:
                raise AttributeError('For system_num = 1, input data should be 2D numpy array.')
        elif self.system_num > 1:
            if type(data) is list and len(data) == self.system_num:
                for ts_data in data:
                    if type(ts_data) is np.ndarray and ts_data.ndim == 2:
                        continue
                    else:
                        raise AttributeError('For system_num > 1, each element of input list should be 2D numpy arrays.')

            else:
                raise AttributeError('For system_num > 1, input data should be the list of `system_num` 2D numpy arrays.')

            # if len(data.shape) != 3:
            #     raise AttributeError('For system_num > 1, input data should have 3 dimensions.')
            # elif self.system_num != data.shape[0]:
            #     raise AttributeError('For system_num > 1, data.shape[0] must equal system_num.')

        return data

    def _forward(self, data, method):
        output_data = []
        for sys_idx, primitive in enumerate(self.primitives):
            sys_data = data[sys_idx]
            sys_data = self._transform(sys_data)
            forward_method = getattr(primitive, method, None)
            output_data.append(forward_method(inputs=sys_data).value.values)
            # print(forward_method(inputs=sys_data).value.values.shape)

        # print(type(output_data), len(output_data), output_data[0].shape)
        # print(np.array(output_data))

        if self.system_num == 1:
            output_data = output_data[0]

        # print(np.array(output_data))

        return output_data

        # output_data = np.array(output_data)
        # if self.system_num == 1:
        #     output_data = output_data.squeeze(axis=0)


    def _transform(self, X):     #transform the ndarray to d3m dataframe, select columns to use
        column_name = [str(col_index) for col_index in range(X.shape[1])]
        return container.DataFrame(X, columns=column_name, generate_metadata=True)

    # def set_training_data(self, data):
    #     return self.primitive.set_training_data(inputs=data)
