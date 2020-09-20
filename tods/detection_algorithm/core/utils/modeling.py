from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten
import numpy as np
import os
import logging

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')


class Model:
    def __init__(self, channel,patience,min_delta,layers,dropout,n_predictions,loss_metric,
                 optimizer,lstm_batch_size,epochs,validation_split,batch_size,l_s
                ):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        # self.config = config
        # self.chan_id = channel.id
        # self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        
        # self.save()

        self._patience = patience
        self._min_delta = min_delta
        self._layers = layers
        self._dropout = dropout
        self._n_predictions = n_predictions
        self._loss_metric = loss_metric
        self._optimizer = optimizer
        self._lstm_batch_size = lstm_batch_size
        self._epochs = epochs
        self._validation_split = validation_split
        self._batch_size = batch_size
        self._l_s = l_s
        
        self.train_new(channel)


    # def load(self):
    #     """
    #     Load model for channel.
    #     """

    #     logger.info('Loading pre-trained model')
    #     self.model = load_model(os.path.join('data', self.config.use_id,
    #                                          'models', self.chan_id + '.h5'))

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self._patience,
                                        min_delta=self._min_delta,
                                        verbose=1)]

        self.model = Sequential()

        self.model.add(LSTM(
            self._layers[0],
            input_shape=(None, channel.X_train.shape[2]),
            return_sequences=True))
        self.model.add(Dropout(self._dropout))

        self.model.add(LSTM(
            self._layers[1],
            return_sequences=False))
        self.model.add(Dropout(self._dropout))

        self.model.add(Dense(
            self._n_predictions
            *channel.X_train.shape[2]
            ))
        self.model.add(Activation('linear'))

        self.model.compile(loss=self._loss_metric,
                           optimizer=self._optimizer)

        
        # print(self.model.summary())

        self.model.fit(channel.X_train,
                       channel.y_train,
                       batch_size=self._lstm_batch_size,
                       epochs=self._epochs,
                       shuffle=False,
                       validation_split=self._validation_split,
                       callbacks=cbs,
                       verbose=True)



    # def save(self):
    #     """
    #     Save trained model.
    #     """

    #     self.model.save(os.path.join('data', self.run_id, 'models',
    #                                  '{}.h5'.format(self.chan_id)))

    def aggregate_predictions(self, y_hat_batch, method='mean'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self._n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)



    def batch_predict(self, channel):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        # num_batches = int((y_test.shape[0] - self._l_s)
        #                   / self._batch_size)
        # if num_batches < 0:
        #     raise ValueError('l_s ({}) too large for stream length {}.'
        #                      .format(self._l_s, y_test.shape[0]))

        # # simulate data arriving in batches, predict each batch
        # for i in range(0, num_batches + 1):
        #     prior_idx = i * self._batch_size
        #     idx = (i + 1) * self._batch_size

        #     if i + 1 == num_batches + 1:
        #         # remaining values won't necessarily equal batch size
        #         idx = y_test.shape[0]

        #     X_test_batch = X_test[prior_idx:idx]
        #     y_hat_batch = self.model.predict(X_test_batch)
        #     y_hat_batch = np.reshape(y_hat_batch,(X_test.shape[0],self._n_predictions,X_test.shape[2]))
        #     # print("PREDICTIONS",y_hat_batch.shape)
        #     self.aggregate_predictions(y_hat_batch)

        # self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        # channel.y_hat = self.y_hat

        # # np.save(os.path.join('data', self.run_id, 'y_hat', '{}.npy'
        # #                      .format(self.chan_id)), self.y_hat)

        # return channel

        self.y_hat = self.model.predict(channel.X_test)        
        self.y_hat = np.reshape(self.y_hat,(channel.X_test.shape[0],self._n_predictions,channel.X_test.shape[2]))
        # print("shape before ",self.y_hat.shape)
        channel.y_hat = self.y_hat
        return channel
