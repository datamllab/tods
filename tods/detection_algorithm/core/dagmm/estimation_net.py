# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class EstimationNet:
    """ Estimation Network

    This network converts input feature vector to softmax probability.
    Bacause loss function for this network is not defined,
    it should be implemented outside of this class.
    """
    def __init__(self, hidden_layer_sizes, activation=tf.nn.relu):
        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of sizes of hidden layers.
            For example, if the sizes are [n1, n2],
            layer sizes of the network are:
            input_size -> n1 -> n2
            (network outputs the softmax probabilities of "n2" layer)
        activation : function
            activation function of hidden layer.
            the funtcion of last layer is softmax function.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def inference(self, z, dropout_ratio=None):
        """ Output softmax probabilities

        Parameters
        ----------
        z : tf.Tensor shape : (n_samples, n_features)
            Data inferenced by this network
        dropout_ratio : tf.Tensor shape : 0-dimension float (optional)
            Specify dropout ratio
            (if None, dropout is not applied)

        Results
        -------
        probs : tf.Tensor shape : (n_samples, n_classes)
            Calculated probabilities
        """
        with tf.variable_scope("EstNet"):
            n_layer = 0
            for size in self.hidden_layer_sizes[:-1]:
                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))
                if dropout_ratio is not None:
                    z = tf.layers.dropout(z, dropout_ratio,
                        name="drop_{}".format(n_layer))

            # Last layer uses linear function (=logits)
            size = self.hidden_layer_sizes[-1]
            logits = tf.layers.dense(z, size, activation=None, name="logits")

            # Softmax output
            output = tf.nn.softmax(logits)

        return output
