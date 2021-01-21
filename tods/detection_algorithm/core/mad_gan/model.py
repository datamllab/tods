#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
# from data_utils import get_batch
from tods.detection_algorithm.core.mad_gan import *
from tods.detection_algorithm.core.mad_gan import data_utils
#import tods.detection_algorithm.core.mad_gan.data_utils as data_utils
import pdb
import json
import sys
from tods.detection_algorithm.core.mad_gan.mod_core_rnn_cell_impl import LSTMCell  # modified to allow initializing bias in lstm

# from tensorflow.contrib.rnn import LSTMCell
#tf.logging.set_verbosity(tf.logging.ERROR)
#import mmd
from tods.detection_algorithm.core.mad_gan import mmd

#from tods.detection_algorithm.core.mad_gan import differential_privacy
from tods.detection_algorithm.core.mad_gan.differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from tods.detection_algorithm.core.mad_gan.differential_privacy.dp_sgd.dp_optimizer import sanitizer
from tods.detection_algorithm.core.mad_gan.differential_privacy.privacy_accountant.tf import accountant

# ------------------------------- #
"""
Most of the models are copied from https://github.com/ratschlab/RGAN
"""

# --- to do with latent space --- #

def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    if use_time:
        print('WARNING: use_time has different semantics')
        sample[:, :, 0] = np.linspace(0, 1.0 / seq_length, num=seq_length)
    return sample

# --- samples for testing ---#

def sample_T(batch_size, batch_idx):
    samples_aaa = np.load('./data/samples_aa.npy')
    num_samples_t = samples_aaa.shape[0]
    labels_aaa = np.load('./data/labels_aa.npy')
    idx_aaa = np.load('./data/idx_aa.npy')
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    T_mb = samples_aaa[start_pos:end_pos, :, :]
    L_mb = labels_aaa[start_pos:end_pos, :, :]
    I_mb = idx_aaa[start_pos:end_pos, :, :]
    return T_mb, L_mb, I_mb, num_samples_t

def sample_TT(batch_size):
    samples_aaa = np.load('./data/samples_aa.npy')
    labels_aaa = np.load('./data/labels_aa.npy')
    idx_aaa = np.load('./data/idx_aa.npy')
    T_indices = np.random.choice(len(samples_aaa), size=batch_size, replace=False)
    T_mb = samples_aaa[T_indices, :, :]
    L_mb = labels_aaa[T_indices, :, :]
    I_mb = idx_aaa[T_indices, :, :]
    return T_mb, L_mb, I_mb

# --- to do with training --- #
def train_epoch(epoch, samples, labels, sess, Z, X, D_loss, G_loss, D_solver, G_solver,
                batch_size, use_time, D_rounds, G_rounds, seq_length,
                latent_dim, num_signals):
    """
    Train generator and discriminator for one epoch.
    """
    # for batch_idx in range(0, int(len(samples) / batch_size) - (D_rounds + (cond_dim > 0) * G_rounds), D_rounds + (cond_dim > 0) * G_rounds):
    for batch_idx in range(0, int(len(samples) / batch_size) - (D_rounds + G_rounds), D_rounds + G_rounds):
        # update the discriminator
        X_mb, Y_mb = data_utils.get_batch(samples, batch_size, batch_idx, labels)
        Z_mb = sample_Z(batch_size, seq_length, latent_dim, use_time)
        for d in range(D_rounds):
            # run the discriminator solver
            _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb})

        # update the generator
        for g in range(G_rounds):
            # run the generator solver
            _ = sess.run(G_solver, feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})

    # at the end, get the loss
    D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb,
                                                                     Z: sample_Z(batch_size, seq_length, latent_dim,
                                                                                 use_time=use_time)})
    D_loss_curr = np.mean(D_loss_curr)
    G_loss_curr = np.mean(G_loss_curr)


    return D_loss_curr, G_loss_curr


def GAN_loss(Z, X, generator_settings, discriminator_settings):

    # normal GAN
    G_sample = generator(Z, **generator_settings)

    D_real, D_logit_real = discriminator(X, **discriminator_settings)

    D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings)

    # Measures the probability error in discrete classification tasks in which each class is independent
    # and not mutually exclusive.
    # logits: predicted labels??

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)

    D_loss = D_loss_real + D_loss_fake


    # G_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), axis=1))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)


    return D_loss, G_loss


def GAN_solvers(D_loss, G_loss, learning_rate, batch_size, total_examples, l2norm_bound, batches_per_lot, sigma, dp=False):
    """
    Optimizers
    """
    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    if dp:
        print('Using differentially private SGD to train discriminator!')
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)
        priv_accountant = accountant.GaussianMomentsAccountant(total_examples)
        clip = True
        l2norm_bound = l2norm_bound / batch_size
        batches_per_lot = 1
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
            priv_accountant,
            [l2norm_bound, clip])

        # the trick is that we need to calculate the gradient with respect to
        # each example in the batch, during the DP SGD step
        D_solver = dp_optimizer.DPGradientDescentOptimizer(learning_rate,
                                                           [eps, delta],
                                                           sanitizer=gaussian_sanitizer,
                                                           sigma=sigma,
                                                           batches_per_lot=batches_per_lot).minimize(D_loss, var_list=discriminator_vars)
    else:
        D_loss_mean_over_batch = tf.reduce_mean(D_loss)
        D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=discriminator_vars)
        priv_accountant = None
    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss_mean_over_batch, var_list=generator_vars)
    return D_solver, G_solver, priv_accountant


# --- to do with the model --- #

def create_placeholders(batch_size, seq_length, latent_dim, num_signals):

    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_signals])
    T = tf.placeholder(tf.float32, [batch_size, seq_length, num_signals])
    return Z, X, T

def generator(z, hidden_units_g, seq_length, batch_size, num_signals, reuse=False, parameters=None, learn_scale=True):

    """
    If parameters are supplied, initialise as such
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_signals],
                                  initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_signals, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer,
                                      trainable=learn_scale)
        # inputs
        inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                        state_is_tuple=True,
                        initializer=lstm_initializer,
                        bias_start=bias_start,
                        reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length] * batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G #out put weighted sum
        #        output_2d = tf.multiply(tf.nn.tanh(logits_2d), scale_out_G)
        output_2d = tf.nn.tanh(logits_2d) # logits operation [-1, 1]
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_signals])

    return output_3d


def discriminator(x, hidden_units_d, seq_length, batch_size, reuse=False, parameters=None, batch_mean=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                                      initializer=tf.truncated_normal_initializer())
            b_out_D = tf.get_variable(name='b_out_D', shape=1,
                                      initializer=tf.truncated_normal_initializer())

        else:
            W_out_D = tf.constant_initializer(value=parameters['discriminator/W_out_D:0'])
            b_out_D = tf.constant_initializer(value=parameters['discriminator/b_out_D:0'])

        # inputs
        inputs = x

        # add the average of the inputs to the inputs (mode collapse?
        if batch_mean:
            mean_over_batch = tf.stack([tf.reduce_mean(x, axis=0)] * batch_size, axis=0)
            inputs = tf.concat([x, mean_over_batch], axis=2)
        """
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d,
                                       state_is_tuple=True,
                                       reuse=reuse)
        """
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_units_d,
                                       state_is_tuple=True,
                                       reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs)
        # logit_final = tf.matmul(rnn_outputs[:, -1], W_final_D) + b_final_D
        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D # output weighted sum
        # real logits or actual output layer?
        # logit is a function that maps probabilities ([0,1]) to ([-inf,inf]) ?

        output = tf.nn.sigmoid(logits) # y = 1 / (1 + exp(-x)). output activation [0, 1]. Probability??
        # sigmoid output ([0,1]), Probability?

    return output, logits


# --- display ----#
def display_batch_progression(j, id_max):
    '''
    See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


# --- to do with saving/loading --- #

def dump_parameters(identifier, sess):
    """
    Save model parmaters to a numpy file
    """
    # dump_path = './experiments/parameters/' + identifier + '.npy'
    dump_path = './tods/detection_algorithm/core/mad_gan/experiments/parameters/' + identifier + '.npy'
    model_parameters = dict()
    for v in tf.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    np.save(dump_path, model_parameters)
    print('Recorded', len(model_parameters), 'parameters to', dump_path)
    return True


def load_parameters(identifier):
    """
    Load parameters from a numpy file
    """
    # load_path = './experiments/plots/parameters/' + identifier + '.npy'
    # load_path = './experiments/plots/parameters/parameters_60/' + identifier + '.npy'
    # load_path = './experiments/parameters/' + identifier + '.npy'
    # load_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = np.load(identifier).item()
    return model_parameters





