import numpy as np
import os
import logging

logger = logging.getLogger('telemanom')


class Channel:
    def __init__(self,n_predictions,l_s):
        # , config, chan_id):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        # self.id = chan_id
        # self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None

        self._n_predictions = n_predictions
        self._l_s = l_s

    def shape_train_data(self, arr):
        # , train=True):
        """Shape raw input streams for ingestion into LSTM. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """
        # print("in shape data")
        # print("arr shape",arr.shape)
        # print("ls",self.config.l_s)
        # print("n_pred",self.config.n_predictions)
        data = []

        for i in range(len(arr) - self._l_s - self._n_predictions):
            data.append(arr[i:i + self._l_s + self._n_predictions])
        data = np.array(data)
        # print("data shape",data.shape)
        # assert len(data.shape) == 3

        # if train:
        #     # np.random.shuffle(data)
        #     self.X_train = data[:, :-self.config.n_predictions, :]
        #     self.y_train = data[:, -self.config.n_predictions:, :]  # telemetry value is at position 0
        #     self.y_train = np.reshape(self.y_train,(self.y_train.shape[0],self.y_train.shape[1]*self.y_train.shape[2]))
        #     print("X train shape",self.X_train .shape)
        #     print("Y train shape",self.y_train .shape)
        # else:
        
        self.X_train = data[:, :-self._n_predictions, :]
        self.y_train = data[:, -self._n_predictions:, :]  # telemetry value is at position 0
        self.y_train = np.reshape(self.y_train,(self.y_train.shape[0],self.y_train.shape[1]*self.y_train.shape[2]))
        

        

    def shape_test_data(self, arr):
        data = []

        for i in range(len(arr) - self._l_s - self._n_predictions):
            data.append(arr[i:i + self._l_s + self._n_predictions])
        data = np.array(data)
        # print("data shape",data.shape)
        self.X_test = data[:, :-self._n_predictions, :]
        self.y_test = data[:, -self._n_predictions:, :]  # telemetry value is at position 0
        self.y_test = np.reshape(self.y_test,(self.y_test.shape[0],self.y_test.shape[1]*self.y_test.shape[2]))


    # def load_data(self):
    #     """
    #     Load train and test data from local.
    #     """
    #     # try:
    #     #     self.train = np.load(os.path.join("data", "train", "{}.npy".format(self.id)))
    #     #     self.test = np.load(os.path.join("data", "test", "{}.npy".format(self.id)))

    #     # except FileNotFoundError as e:
    #     #     # logger.critical(e)
    #     #     # logger.critical("Source data not found, may need to add data to repo: <link>")
    #     #     print("Source data not found, may need to add data to repo: <link>")

    #     print("before shape function")
    #     print(self.train.shape)
    #     self.shape_data(self.train)
    #     self.shape_data(self.test, train=False)