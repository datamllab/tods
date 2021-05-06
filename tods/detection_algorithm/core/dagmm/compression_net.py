import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class CompressionNet:
    """ Compression Network.
    This network converts the input data to the representations
    suitable for calculation of anormaly scores by "Estimation Network".

    Outputs of network consist of next 2 components:
    1) reduced low-dimensional representations learned by AutoEncoder.
    2) the features derived from reconstruction error.
    """
    def __init__(self, hidden_layer_sizes, activation=tf.nn.tanh):
        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of the size of hidden layers.
            For example, if the sizes are [n1, n2],
            the sizes of created networks are:
            input_size -> n1 -> n2 -> n1 -> input_sizes
            (network outputs the representation of "n2" layer)
        activation : function
            activation function of hidden layer.
            the last layer uses linear function.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def compress(self, x):
        self.input_size = x.shape[1]

        with tf.variable_scope("Encoder"):
            z = x
            n_layer = 0
            for size in self.hidden_layer_sizes[:-1]:
                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))

            # activation function of last layer is linear
            n_layer += 1
            z = tf.layers.dense(z, self.hidden_layer_sizes[-1],
                name="layer_{}".format(n_layer))

        return z

    def reverse(self, z):
        with tf.variable_scope("Decoder"):
            n_layer = 0
            for size in self.hidden_layer_sizes[:-1][::-1]:
                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))

            # activation function of last layes is linear
            n_layer += 1
            x_dash = tf.layers.dense(z, self.input_size,
                name="layer_{}".format(n_layer))

        return x_dash

    def loss(self, x, x_dash):
        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

        # Calculate Euclid norm, distance
        norm_x = euclid_norm(x)
        norm_x_dash = euclid_norm(x_dash)
        dist_x = euclid_norm(x - x_dash)
        dot_x = tf.reduce_sum(x * x_dash, axis=1)

        # Based on the original paper, features of reconstraction error
        # are composed of these loss functions:
        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity
        min_val = 1e-3
        loss_E = dist_x  / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))
        return tf.concat([loss_E[:,None], loss_C[:,None]], axis=1)

    def extract_feature(self, x, x_dash, z_c):
        z_r = self.loss(x, x_dash)
        return tf.concat([z_c, z_r], axis=1)

    def inference(self, x):
        """ convert input to output tensor, which is composed of
        low-dimensional representation and reconstruction error.

        Parameters
        ----------
        x : tf.Tensor shape : (n_samples, n_features)
            Input data

        Results
        -------
        z : tf.Tensor shape : (n_samples, n2 + 2)
            Result data
            Second dimension of this data is equal to
            sum of compressed representation size and
            number of loss function (=2)

        x_dash : tf.Tensor shape : (n_samples, n_features)
            Reconstructed data for calculation of
            reconstruction error.
        """

        with tf.variable_scope("CompNet"):
            # AutoEncoder
            z_c = self.compress(x)
            x_dash = self.reverse(z_c)

            # compose feature vector
            z = self.extract_feature(x, x_dash, z_c)

        return z, x_dash

    def reconstruction_error(self, x, x_dash):
        return tf.reduce_mean(tf.reduce_sum(
            tf.square(x - x_dash), axis=1), axis=0)
