import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from  .compression_net import CompressionNet
from .estimation_net import EstimationNet
from .gmm import GMM
from pyod.utils.stat_models import pairwise_distances_no_broadcast

from os import makedirs
from os.path import exists, join
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pyod.models.base import BaseDetector

class DAGMM(BaseDetector):
    """ Deep Autoencoding Gaussian Mixture Model.

    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens:list = [16,8,1],
            est_hiddens:list = [8,4], est_dropout_ratio:float =0.5,
            minibatch_size:int = 1024, epoch_size:int =100,
            learning_rate:float =0.0001, lambda1:float =0.1, lambda2:float =0.0001,
            normalize:bool=True, random_seed:int=123 , contamination:float = 0.001  ):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes

        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)

        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        normalize : bool (optional)
            specify whether input data need to be normalized.
            by default, input data is normalized.
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        est_activation = tf.nn.tanh
        comp_activation = tf.nn.tanh
        super(DAGMM, self).__init__(contamination=contamination)
        self.comp_net = CompressionNet(comp_hiddens, comp_activation)
        self.est_net = EstimationNet(est_hiddens, est_activation)
        self.est_dropout_ratio = est_dropout_ratio

        n_comp = est_hiddens[-1]
        self.gmm = GMM(n_comp)

        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed

        self.graph = None
        self.sess = None

    #def __del__(self):
     #   if self.sess is not None:
      #      self.sess.close()

    def fit(self,X,y=None):
        """ Fit the DAGMM model according to the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        """

        n_samples, n_features = X.shape

        if self.normalize:
            self.scaler = scaler = StandardScaler()
            X = scaler.fit_transform(X)

        with tf.Graph().as_default() as graph:
            self.graph = graph
            tf.set_random_seed(self.seed)
            np.random.seed(seed=self.seed)

            # Create Placeholder
            self.input = input = tf.placeholder(
                dtype=tf.float32, shape=[None, n_features])
            self.drop = drop = tf.placeholder(dtype=tf.float32, shape=[])

            # Build graph
            z, x_dash  = self.comp_net.inference(input)
            gamma = self.est_net.inference(z, drop)
            self.gmm.fit(z, gamma)
            energy = self.gmm.energy(z)

            self.x_dash = x_dash

            # Loss function
            loss = (self.comp_net.reconstruction_error(input, x_dash) +
                self.lambda1 * tf.reduce_mean(energy) +
                self.lambda2 * self.gmm.cov_diag_loss())

            # Minimizer
            minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            # Number of batch
            n_batch = (n_samples - 1) // self.minibatch_size + 1

            # Create tensorflow session and initilize
            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)
            self.sess.run(init)

            # Training
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)

            for epoch in range(self.epoch_size):
                for batch in range(n_batch):
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = X[idx[i_start:i_end]]

                    self.sess.run(minimizer, feed_dict={
                        input:x_batch, drop:self.est_dropout_ratio})
                if (epoch + 1) % 10 == 0:
                    loss_val = self.sess.run(loss, feed_dict={input:X, drop:0})
                    print(" epoch {}/{} : loss = {:.3f}".format(epoch + 1, self.epoch_size, loss_val))

            # Fix GMM parameter
            fix = self.gmm.fix_op()
            self.sess.run(fix, feed_dict={input:X, drop:0})
            self.energy = self.gmm.energy(z)

            tf.add_to_collection("save", self.input)
            tf.add_to_collection("save", self.energy)

            self.saver = tf.train.Saver()

            pred_scores = self.decision_function(X)
            self.decision_scores_ = pred_scores
            self._process_decision_scores()
            #return self

    def decision_function(self, X):
        """ Calculate anormaly scores (sample energy) on samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.

        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if self.normalize:
            X = self.scaler.transform(X)

        energies = self.sess.run(self.energy, feed_dict={self.input:X})

        return energies.reshape(1,-1)

    def save(self, fdir):
        """ Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if not exists(fdir):
            makedirs(fdir)

        model_path = join(fdir, self.MODEL_FILENAME)
        self.saver.save(self.sess, model_path)

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            joblib.dump(self.scaler, scaler_path)

    def restore(self, fdir):
        """ Restore trained model from designated directory.

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        if not exists(fdir):
            raise Exception("Model directory does not exist.")

        model_path = join(fdir, self.MODEL_FILENAME)
        meta_path = model_path + ".meta"

        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.sess = tf.Session(graph=graph)
            self.saver = tf.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, model_path)

            self.input, self.energy = tf.get_collection("save")

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            self.scaler = joblib.load(scaler_path)
