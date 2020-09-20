from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy as np
import typing
import pandas as pd


from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten

from d3m import container, utils
from d3m.base import utils as base_ut
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase

from .core.CollectiveBase import CollectiveBaseDetector

from sklearn.utils import check_array

# from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions


# from detection_algorithm.UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase

from .core.utils.errors import Errors
from .core.utils.channel import Channel
from .core.utils.modeling import Model

# from pyod.models.base import BaseDetector



__all__ = ('Telemanom',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(Params_ODBase):
    ######## Add more Attributes #######

        pass


class Hyperparams(Hyperparams_ODBase):	


    smoothing_perc = hyperparams.Hyperparameter[float](
            default=0.05,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="determines window size used in EWMA smoothing (percentage of total values for channel)"
            )


    window_size_ = hyperparams.Hyperparameter[int](
            default=100,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="number of trailing batches to use in error calculation"
            )

    error_buffer = hyperparams.Hyperparameter[int](
            default=50,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="number of values surrounding an error that are brought into the sequence (promotes grouping on nearby sequences"
            )

    batch_size = hyperparams.Hyperparameter[int](
            default=70,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Batch size while predicting"
            )


    # LSTM Model Parameters
    dropout = hyperparams.Hyperparameter[float](
            default=0.3,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="Dropout rate"
            )

    validation_split = hyperparams.Hyperparameter[float](
            default=0.2,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Validation split"
            )

    optimizer = hyperparams.Hyperparameter[typing.Union[str, None]](
            default='Adam',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Optimizer"
            )


    lstm_batch_size = hyperparams.Hyperparameter[int](
            default=64,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="lstm model training batch size"
            )


    loss_metric = hyperparams.Hyperparameter[typing.Union[str, None]](
            default='mean_squared_error',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="loss function"
            )


    layers = hyperparams.List(
            elements=hyperparams.Hyperparameter[int](1),
            default=[10,10],
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="No of units for the 2 lstm layers"
            )

    # Training Parameters

    epochs = hyperparams.Hyperparameter[int](
            default=1,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="Epoch"
            )

    patience  = hyperparams.Hyperparameter[int](
            default=10,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="Number of consequetive training iterations to allow without decreasing the val_loss by at least min_delta"
            )

    min_delta = hyperparams.Hyperparameter[float](
            default=0.0003,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="Number of consequetive training iterations to allow without decreasing the val_loss by at least min_delta"
            )


    l_s = hyperparams.Hyperparameter[int](
            default=100,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="num previous timesteps provided to model to predict future values"
            )

    n_predictions = hyperparams.Hyperparameter[int](
            default=10,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
            description="number of steps ahead to predict"
            )


    # Error thresholding parameters
    # ==================================

    p = hyperparams.Hyperparameter[float](
            default=0.05,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            description="minimum percent decrease between max errors in anomalous sequences (used for pruning)"
            )

    # Contamination

    contamination = hyperparams.Uniform(
            lower=0.,
            upper=0.5,
            default=0.1,
            description='the amount of contamination of the data set, i.e.the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function',
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )



class TelemanomPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that uses telmanom for outlier detection

    Parameters
            ----------


    """

    __author__ = "Data Lab"
    metadata = metadata_base.PrimitiveMetadata(
    {
            '__author__' : "DATA Lab at Texas A&M University",
            'name': "Telemanom",
            'python_path': 'd3m.primitives.tods.detection_algorithm.telemanom',
            'source': {
            'name': 'DATA Lab at Texas A&M University',
            'contact': 'mailto:khlai037@tamu.edu',
            'uris': [
                    'https://gitlab.com/lhenry15/tods.git',
                    'https://gitlab.com/lhenry15/tods/-/blob/purav/anomaly-primitives/anomaly_primitives/telemanom.py',
            ],
            },
            'algorithm_types': [
                    metadata_base.PrimitiveAlgorithmType.TELEMANOM,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
            'id': 'c7259da6-7ce6-42ad-83c6-15238679f5fa',
            'hyperparameters_to_tune':['layers','loss_metric','optimizer','epochs','p','l_s','patience','min_delta','dropout','smoothing_perc'],
            'version': '0.0.1',
    },
    )

    def __init__(self, *,
                             hyperparams: Hyperparams,  #
                             random_seed: int = 0,
                             docker_containers: Dict[str, DockerContainer] = None) -> None:

            super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

            self._clf = Detector(smoothing_perc=self.hyperparams['smoothing_perc'],
                                            window_size=self.hyperparams['window_size_'],
                                            error_buffer=self.hyperparams['error_buffer'],
                                            batch_size = self.hyperparams['batch_size'],
                                            validation_split = self.hyperparams['validation_split'],
                                            optimizer = self.hyperparams['optimizer'],
                                            lstm_batch_size = self.hyperparams['lstm_batch_size'],
                                            loss_metric = self.hyperparams['loss_metric'],
                                            layers = self.hyperparams['layers'],
                                            epochs = self.hyperparams['epochs'],
                                            patience = self.hyperparams['patience'],
                                            min_delta = self.hyperparams['min_delta'],
                                            l_s = self.hyperparams['l_s'],
                                            n_predictions = self.hyperparams['n_predictions'],
                                            p = self.hyperparams['p'],
                                            contamination=hyperparams['contamination']
                                            )

    def set_training_data(self, *, inputs: Inputs) -> None:
            """
            Set training data for outlier detection.
            Args:
                    inputs: Container DataFrame

            Returns:
                    None
            """
            super().set_training_data(inputs=inputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
            """
            Fit model with training data.
            Args:
                    *: Container DataFrame. Time series data up to fit.

            Returns:
                    None
            """
            return super().fit()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
            """
            Process the testing data.
            Args:
                    inputs: Container DataFrame. Time series data up to outlier detection.

            Returns:
                    Container DataFrame
                    1 marks Outliers, 0 marks normal.
            """
            return super().produce(inputs=inputs, timeout=timeout, iterations=iterations)


    def produce_score(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.
        Returns:
            Container DataFrame
            Outlier score of input DataFrame.
        """
        return super().produce_score(inputs=inputs, timeout=timeout, iterations=iterations)


    def get_params(self) -> Params:
            """
            Return parameters.
            Args:
                    None

            Returns:
                    class Params
            """
            return super().get_params()

    def set_params(self, *, params: Params) -> None:
            """
            Set parameters for outlier detection.
            Args:
                    params: class Params

            Returns:
                    None
            """
            super().set_params(params=params)



class Detector(CollectiveBaseDetector):
        """Class to Implement Deep Log LSTM based on "https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
           Only Parameter Value anomaly detection layer has been implemented for time series data"""

        def __init__(self,smoothing_perc=0.05,window_size = 10,error_buffer = 5,batch_size =30, \
                dropout = 0.3, validation_split=0.2,optimizer='adam',lstm_batch_size=64,loss_metric='mean_squared_error', \
                layers=[40,40],epochs = 1,patience =10,min_delta=0.0003,l_s=5,n_predictions=2,p = 0.05,contamination=0.1):

                # super(Detector, self).__init__(contamination=contamination)
                super(Detector, self).__init__(contamination=contamination,
                                                                                          window_size=l_s,
                                                                                          step_size=1,
                                                                                          )

                self._smoothin_perc = smoothing_perc
                self._window_size =window_size
                self._error_buffer = error_buffer
                self._batch_size = batch_size
                self._dropout = dropout
                self._validation_split = validation_split
                self._optimizer = optimizer
                self._lstm_batch_size = lstm_batch_size
                self._loss_metric = loss_metric
                self._layers = layers
                self._epochs = epochs
                self._patience = patience
                self._min_delta = min_delta
                self._l_s = l_s
                self._n_predictions = n_predictions
                self._p = p
                self.contamination = contamination

                # self.y_hat = None
                self.results = []
                self.result_df = None

                self._model = None
                self._channel = None


        def fit(self,X,y=None):
                """
                Fit data to  LSTM model.
                Args:
                        inputs : X , ndarray of size (number of sample,features)

                Returns:
                        return : self object with trained model
                """
                X = check_array(X).astype(np.float)
                self._set_n_classes(None)

                inputs = X
                self._channel = Channel(n_predictions = self._n_predictions,l_s = self._l_s)
                self._channel.shape_train_data(inputs)

                self._model = Model(self._channel,patience = self._patience,
                                                          min_delta =self._min_delta,
                                                          layers = self._layers,
                                                          dropout = self._dropout,
                                                          n_predictions = self._n_predictions, 
                                                          loss_metric = self._loss_metric,
                                                          optimizer = self._optimizer,
                                                          lstm_batch_size = self._lstm_batch_size,
                                                          epochs = self._epochs,
                                                          validation_split = self._validation_split,
                                                          batch_size = self._batch_size,
                                                          l_s = self._l_s
                                                )

                self.decision_scores_, self.left_inds_, self.right_inds_ = self.decision_function(X)
                self._process_decision_scores()

                return self



        def decision_function(self, X: np.array):
                """Predict raw anomaly scores of X using the fitted detector.

                The anomaly score of an input sample is computed based on the fitted
                detector. For consistency, outliers are assigned with
                higher anomaly scores.

                Parameters
                ----------
                X : numpy array of shape (n_samples, n_features)
                        The input samples. Sparse matrices are accepted only
                        if they are supported by the base estimator.

                Returns
                -------
                anomaly_scores : numpy array of shape (n_samples,)
                        The anomaly score of the input samples.
                """

                X = check_array(X).astype(np.float)
                self._set_n_classes(None)

                inputs = X
                self._channel.shape_test_data(inputs)
                self._channel = self._model.batch_predict(channel = self._channel)

                errors = Errors(channel = self._channel,
                                                window_size = self._window_size,
                                                batch_size = self._batch_size,
                                                smoothing_perc = self._smoothin_perc,
                                                n_predictions = self._n_predictions, 
                                                l_s = self._l_s,
                                                error_buffer = self._error_buffer,
                                                p = self._p
                                                )

                # prediciton smoothed error
                prediction_errors = np.reshape(errors.e_s,(self._channel.X_test.shape[0],self._channel.X_test.shape[2]))
                prediction_errors = np.sum(prediction_errors,axis=1)

                left_indices = []
                right_indices = []
                scores = []
                for i in range(len(prediction_errors)):
                        left_indices.append(i)
                        right_indices.append(i+self._l_s)
                        scores.append(prediction_errors[i])



                return np.asarray(scores),np.asarray(left_indices),np.asarray(right_indices)



# if __name__ == "__main__":

# 	csv  = pd.read_csv("/home/purav/Downloads/yahoo_train.csv")
# 	# X_train = np.asarray(
# 	#     [3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]).reshape(-1, 1)

# 	# X_test = np.asarray(
# 	#     [3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]).reshape(-1,1)

# 	# print(X_train.shape, X_test.shape)

# 	X_train  = csv.iloc[:,[2,3,4,5,6]].values

# 	clf = Detector(contamination=0.1)
# 	clf.fit(X_train)
# 	# pred_scores = clf.decision_function(X_test)
# 	pred_labels = clf.predict(X_train)

# 	print(clf.threshold_)
# 	# print(np.percentile(pred_scores, 100 * 0.9))

# 	# print('pred_scores: ',pred_scores)
# 	print('scores: ',pred_labels[0].shape)
# 	print('left_indices: ',pred_labels[1].shape)
# 	print('right_indices: ',pred_labels[2].shape)
