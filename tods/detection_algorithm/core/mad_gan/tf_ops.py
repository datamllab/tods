### from https://github.com/eugenium/MMD/blob/master/tf_ops.py
import tensorflow as tf


def sq_sum(t, name=None):
    "The squared Frobenius-type norm of a tensor, sum(t ** 2)."
    with tf.name_scope(name, "SqSum", [t]):
        t = tf.convert_to_tensor(t, name='t')
        return 2 * tf.nn.l2_loss(t)


def dot(x, y, name=None):
    "The dot product of two vectors x and y."
    with tf.name_scope(name, "Dot", [x, y]):
        x = tf.convert_to_tensor(x, name='x')
        y = tf.convert_to_tensor(y, name='y')

        x.get_shape().assert_has_rank(1)
        y.get_shape().assert_has_rank(1)

        return tf.squeeze(tf.matmul(tf.expand_dims(x, 0), tf.expand_dims(y, 1)))
