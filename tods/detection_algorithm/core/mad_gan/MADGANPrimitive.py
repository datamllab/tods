from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout , LSTM
from keras.regularizers import l2
from keras.losses import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.models.base import BaseDetector

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
# from numba import njit
from pyod.utils.utility import argmaxn

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas
import uuid

from d3m import container, utils as d3m_utils

from tods.detection_algorithm.UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase

# from RGAN.py
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pdb
import random
import json
from scipy.stats import mode

from tods.detection_algorithm.core.mad_gan import *
import data_utils
import plotting
import model
import utils
import eval
import DR_discriminator

# import tods.detection_algorithm.core.mad_gan.data_utils
# import tods.detection_algorithm.core.mad_gan.plotting
# import tods.detection_algorithm.core.mad_gan.model
# import tods.detection_algorithm.core.mad_gan.utils
# import tods.detection_algorithm.core.mad_gan.eval
# import tods.detection_algorithm.core.mad_gan.DR_discriminator

from time import time
from math import floor
from mmd import rbf_mmd2, median_pairwise_distance, mix_rbf_mmd2_and_ratio

# from AD.py
from mod_core_rnn_cell_impl import LSTMCell
from sklearn.utils.validation import *
from sklearn.metrics.classification import *
from sklearn.metrics.ranking import *
from time import time

begin = time()


__all__ = ('MADGANPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(Params_ODBase):
    ######## Add more Attributes #######

    pass


class Hyperparams(Hyperparams_ODBase):
    # options pertaining to data
    seq_length = hyperparams.Hyperparameter[int](
        default=30,
        description='Selecting a suitable subsequence resolution (ie. sub-sequence length)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    num_signals = hyperparams.Hyperparameter[int](
        default=1,
        description='Number of Signals',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    normalise = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="normalise the training/vali/test data (during split)?",
    )
    cond_dim = hyperparams.Hyperparameter[int](
        default=0,
        description='dimension of *conditional* input',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_val = hyperparams.Hyperparameter[int](
        default=0,
        description='assume conditional codes come from [0, max_val)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    one_hot = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="convert categorical conditional information to one-hot encoding",
    )
    predict_labels = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="instead of conditioning with labels, require model to output them",
    )

    # hyperparameters of the model
    hidden_units_g = hyperparams.Hyperparameter[int](
        default=1000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    hidden_units_d = hyperparams.Hyperparameter[int](
        default=1000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    kappa = hyperparams.Uniform(
        lower=0,
        upper=1.0,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="weight between final output and intermediate steps in discriminator cost (1 = all intermediate",
    )
    latent_dim = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="dimensionality of the latent/noise space",
    )
    batch_mean = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="append the mean of the batch to all variables for calculating discriminator loss",
    )
    learn_scale = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="make the 'scale' parameter at the output of the generator learnable (else fixed to 1",
    )

    # options pertaining to training
    learning_rate = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=28,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    num_epochs = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    D_rounds = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="number of rounds of discriminator training",
    )
    G_rounds = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="number of rounds of generator training",
    )
    use_time = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="enforce latent dimension 0 to correspond to time",
    )
    WGAN = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    WGAN_clip = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    shuffle = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    wrong_labels = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="augment discriminator loss with real examples with wrong (~shuffled, sort of) labels",
    )

    # options pertaining to evaluation and exploration
    identifier = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="identifier string for output files",
    )

    # options pertaining to differential privacy
    dp = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="train discriminator with differentially private SGD?",
    )
    l2norm_bound = hyperparams.Uniform(
        lower=0,
        upper=10000,
        default=1e-5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="bound on norm of individual gradients for DP training",
    )
    batches_per_lot = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="number of batches per lot (for DP)",
    )
    dp_sigma = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1e-5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="sigma for noise added (for DP)",
    )


class MADGANPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that uses MAD-GAN for outlier detection

    Parameters
        ----------


    """

    __author__ = "DATA Lab at Texas A&M University",
    metadata = metadata_base.PrimitiveMetadata(
        {
        '__author__': "DATA Lab @Texas A&M University",
        'name': "DeepLog Anomolay Detection",
        'python_path': 'd3m.primitives.tods.detection_algorithm.deeplog',
        'source': {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                   'uris': ['https://gitlab.com/lhenry15/tods/-/blob/Yile/anomaly-primitives/anomaly_primitives/MatrixProfile.py']},
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.DEEPLOG], 
        'primitive_family': metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
        'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'DeepLogPrimitive')),
        'hyperparams_to_tune': [],
        'version': '0.0.1', 
        }
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,  #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._clf = MADGAN(hidden_units_g=hyperparams['hidden_units_g'],
                           hidden_units_d=hyperparams['hidden_units_d'],
                           kappa=hyperparams['kappa'],
                           latent_dim=hyperparams['latent_dim'],
                           batch_mean=hyperparams['batch_mean'],
                           learn_scale=hyperparams['learn_scale'],
                           learning_rate=hyperparams['learning_rate'],
                           batch_size=hyperparams['batch_size'],
                           num_epochs=hyperparams['num_epochs'],
                           D_rounds=hyperparams['D_rounds'],
                           G_rounds=hyperparams['G_rounds'],
                           use_time=hyperparams['use_time'],
                           WGAN=hyperparams['WGAN'],
                           WGAN_clip = hyperparams['WGAN_clip'],
                           shuffle = hyperparams['shuffle'],
                           wrong_labels = hyperparams['wrong_labels'])
                                

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

class myADclass():
    def __init__(self, epoch, settings=settings, samples=samples, labels=labels, index=index):
        self.epoch = epoch
        self.settings = settings
        self.samples = samples
        self.labels = labels
        self.index = index

    def ADfunc(self):
        num_samples_t = self.samples.shape[0]
        print('sample_shape:', self.samples.shape[0])
        print('num_samples_t', num_samples_t)

        # -- only discriminate one batch for one time -- #
        D_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        DL_test = np.empty([num_samples_t, self.settings['seq_length'], 1])
        L_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        I_mb = np.empty([num_samples_t, self.settings['seq_length'], 1])
        batch_times = num_samples_t // self.settings['batch_size']
        for batch_idx in range(0, num_samples_t // self.settings['batch_size']):
            # print('batch_idx:{}
            # display batch progress
            model.display_batch_progression(batch_idx, batch_times)
            start_pos = batch_idx * self.settings['batch_size']
            end_pos = start_pos + self.settings['batch_size']
            T_mb = self.samples[start_pos:end_pos, :, :]
            L_mmb = self.labels[start_pos:end_pos, :, :]
            I_mmb = self.index[start_pos:end_pos, :, :]
            para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
                self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
            D_t, L_t = DR_discriminator.dis_trained_model(self.settings, T_mb, para_path)
            D_test[start_pos:end_pos, :, :] = D_t
            DL_test[start_pos:end_pos, :, :] = L_t
            L_mb[start_pos:end_pos, :, :] = L_mmb
            I_mb[start_pos:end_pos, :, :] = I_mmb

        start_pos = (num_samples_t // self.settings['batch_size']) * self.settings['batch_size']
        end_pos = start_pos + self.settings['batch_size']
        size = samples[start_pos:end_pos, :, :].shape[0]
        fill = np.ones([self.settings['batch_size'] - size, samples.shape[1], samples.shape[2]])
        batch = np.concatenate([samples[start_pos:end_pos, :, :], fill], axis=0)
        para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
            self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
        D_t, L_t = DR_discriminator.dis_trained_model(self.settings, batch, para_path)
        L_mmb = self.labels[start_pos:end_pos, :, :]
        I_mmb = self.index[start_pos:end_pos, :, :]
        D_test[start_pos:end_pos, :, :] = D_t[:size, :, :]
        DL_test[start_pos:end_pos, :, :] = L_t[:size, :, :]
        L_mb[start_pos:end_pos, :, :] = L_mmb
        I_mb[start_pos:end_pos, :, :] = I_mmb

        results = np.zeros([18, 4])
        for i in range(2, 8):
            tao = 0.1 * i
            Accu2, Pre2, Rec2, F12 = DR_discriminator.detection_Comb(
                DL_test, L_mb, I_mb, self.settings['seq_step'], tao)
            print('seq_length:', self.settings['seq_length'])
            print('Comb-logits-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(self.epoch, tao, Accu2, Pre2, Rec2, F12))
            results[i - 2, :] = [Accu2, Pre2, Rec2, F12]

            Accu3, Pre3, Rec3, F13 = DR_discriminator.detection_Comb(
                D_test, L_mb, I_mb, self.settings['seq_step'], tao)
            print('seq_length:', self.settings['seq_length'])
            print('Comb-statistic-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(self.epoch, tao, Accu3, Pre3, Rec3, F13))
            results[i - 2+6, :] = [Accu3, Pre3, Rec3, F13]

            Accu5, Pre5, Rec5, F15 = DR_discriminator.sample_detection(D_test, L_mb, tao)
            print('seq_length:', self.settings['seq_length'])
            print('sample-wise-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}'
                  .format(self.epoch, tao, Accu5, Pre5, Rec5, F15))
            results[i - 2+12, :] = [Accu5, Pre5, Rec5, F15]

        return results

class MADGAN(BaseDetector):
    """Class to Implement Deep Log LSTM based on "https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
       Only Parameter Value anomaly detection layer has been implemented for time series data"""

    def __init__(self, settings_file: str= "", data: str="kdd99", seq_length: int=30, num_signals: int=6,
                    normalise: int=false, scale: float=0.1, freq_low: float=1.0, freq_high: float=5.0,
                    amplitude_low: float=0.1, amplitude_high: float=0.9, multivariate_mnist: bool=false,
                    full_mnist: bool=false, data_load_from: str="", resample_rate_in_min: int=15,
                    hidden_units_g: int=100, hidden_units_d: int=100, hidden_units_e: int=100,
                    kappa: int=1, latent_dim: int=15, weight: float=0.5, degree: int=1, batch_mean: bool=false,
                    learn_scale: bool=false, learning_rate: float=0.1, batch_size: float=500, num_epochs: float=100,
                    D_rounds: int=1, G_rounds: int=3, E_rounds: int=1, shuffle: bool=true, eval_mul: bool=false,
                    eval_an: bool=false, eval_single: bool=false, wrong_labels: bool=false, identifier: str="kdd99",
                    sub_id: str="kdd99", dp: bool=false, l2norm_bound: float=1e-05, batches_per_lot: int=1,
                    dp_sigma: float=1e-05, use_time: bool=false, seq_step: int=10, num_generated_features: int=6):

        super(DeeplogLstm, self).__init__(contamination=contamination)
        self.MG_hyperparams['hidden_units_g'] = hidden_units_g
        self.MG_hyperparams['hidden_units_d'] = hidden_units_d
        self.MG_hyperparams['kappa'] = kappa
        self.MG_hyperparams['latent_dim'] = latent_dim
        self.MG_hyperparams['batch_mean'] = batch_mean
        self.MG_hyperparams['learn_scale'] = learn_scale

        self.MG_hyperparams['learning_rate']  = learning_rateS
        self.MG_hyperparams['batch_size']  = batch_size 
        self.MG_hyperparams['num_epochs']  = num_epochs
        self.MG_hyperparams['D_rounds']  = D_rounds
        self.MG_hyperparams['G_rounds']  = G_rounds
        self.MG_hyperparams['use_time']  = use_time
        self.MG_hyperparams['WGAN']  = WGANS
        self.MG_hyperparams['WGAN_clip']  = WGAN_clip
        self.MG_hyperparams['shuffle']  = shuffle
        self.MG_hyperparams['wrong_labels']  = wrong_labels
        
        self.MG_hyperparams['data'] = data
        self.MG_hyperparams['seq_length'] = seq_length
        self.MG_hyperparams['seq_step'] = seq_step
        self.MG_hyperparams['num_signals'] = num_signals
        self.MG_hyperparams['sub_id'] = sub_id
        self.MG_hyperparams['evan_an'] = eval_an


    def _build_model(self):
        """
        Builds Stacked LSTM model.
        Args:
            inputs : Self object containing model parameters

        Returns:
            return : model
        """
        # taken from RGAN.py 
        begin = time()

        #tf.logging.set_verbosity(tf.logging.ERROR)
        # --- get settings --- #
        # parse command line arguments, or use defaults
        parser = utils.rgan_options_parser()
        settings = vars(parser.parse_args())
        # if a settings file is specified, it overrides command line arguments/defaults
        if settings['settings_file']: settings = utils.load_settings_from_file(settings)
        # --- get data, split --- #
        # samples, pdf, labels = data_utils.get_data(settings)
        data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
        print('Loading data from', data_path)
        settings["eval_an"] = False
        settings["eval_single"] = False
        samples, labels, index = data_utils.get_data(settings["data"], settings["seq_length"], settings["seq_step"],
                                                     settings["num_signals"], settings['sub_id'], settings["eval_single"],
                                                     settings["eval_an"], data_path)
        print('samples_size:',samples.shape)
        # -- number of variables -- #
        num_variables = samples.shape[2]
        print('num_variables:', num_variables)
        # --- save settings, data --- #
        print('Ready to run with settings:')
        for (k, v) in settings.items(): print(v, '\t', k)
        # add the settings to local environment
        # WARNING: at this point a lot of variables appear
        locals().update(settings)
        json.dump(settings, open('./tods/detection_algorithm/core/mad_gan/experiments/settings/' + identifier + '.txt', 'w'), indent=0)
        # --- build model --- #
        # preparation: data placeholders and model parameters
        Z, X, T = model.create_placeholders(batch_size, seq_length, latent_dim, num_variables)
        discriminator_vars = ['hidden_units_d', 'seq_length', 'batch_size', 'batch_mean']
        discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
        generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'learn_scale']
        generator_settings = dict((k, settings[k]) for k in generator_vars)
        generator_settings['num_signals'] = num_variables
        # model: GAN losses
        D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings)
        D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size,
                                                                total_examples=samples.shape[0],
                                                                l2norm_bound=l2norm_bound,
                                                                batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
        # model: generate samples for visualization
        G_sample = model.generator(Z, **generator_settings, reuse=True)

        

    def fit(self,X,y=None):
        """
        Fit data to  LSTM model.
        Args:
            inputs : X , ndarray of size (number of sample,features)

        Returns:
            return : self object with trained model
        """
        
        self._build_model()

        # --- run the program --- #
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # # -- plot the real samples -- #
        vis_real_indices = np.random.choice(len(samples), size=16)
        vis_real = np.float32(samples[vis_real_indices, :, :])
        plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=16, num_epochs=num_epochs)
        plotting.save_samples_real(vis_real, identifier)

        # --- train --- #
        train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 'latent_dim']
        train_settings = dict((k, settings[k]) for k in train_vars)
        train_settings['num_signals'] = num_variables

        t0 = time()
        MMD = np.zeros([num_epochs, ])

        for epoch in range(num_epochs):
        # for epoch in range(1):
            # -- train epoch -- #
            D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples, labels, sess, Z, X, D_loss, G_loss,
                                                         D_solver, G_solver, **train_settings)

            # -- print -- #
            print('epoch, D_loss_curr, G_loss_curr, seq_length')
            print('%d\t%.4f\t%.4f\t%d' % (epoch, D_loss_curr, G_loss_curr, seq_length))


            # -- save model parameters -- #
            model.dump_parameters(sub_id + '_' + str(seq_length) + '_' + str(epoch), sess)

        np.save('./tods/detection_algorithm/core/mad_gan/experiments/plots/gs/' + identifier + '_' + 'MMD.npy', MMD)

        end = time() - begin
        print('Training terminated | Training time=%d s' %(end) )

        print("Training terminated | training time = %ds  " % (time() - begin))


       
    def produce(self, X):
        # --- get settings --- #
        # parse command line arguments, or use defaults
        parser = utils.rgan_options_parser()
        settings = vars(parser.parse_args())
        # if a settings file is specified, it overrides command line arguments/defaults
        if settings['settings_file']: settings = utils.load_settings_from_file(settings)

        # --- get data, split --- #
        data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
        print('Loading data from', data_path)
        settings["eval_single"] = False
        settings["eval_an"] = False
        samples, labels, index = data_utils.get_data(settings["data"], settings["seq_length"], settings["seq_step"],
                                                     settings["num_signals"], settings["sub_id"], settings["eval_single"],
                                                     settings["eval_an"], data_path)
        # --- save settings, data --- #
        # no need
        print('Ready to run with settings:')
        for (k, v) in settings.items(): print(v, '\t', k)
        # add the settings to local environment
        # WARNING: at this point a lot of variables appear
        locals().update(settings)
        json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

        print('Main Starting...')

        Results = np.empty([settings['num_epochs'], 18, 4])

        for epoch in range(settings['num_epochs']):
        # for epoch in range(50, 60):
            ob = myADclass(epoch)
            Results[epoch, :, :] = ob.ADfunc()

        # res_path = './experiments/plots/Results' + '_' + settings['sub_id'] + '_' + str(
        #     settings['seq_length']) + '.npy'
        # np.save(res_path, Results)

        print('Main Terminating...')
        end = time() - begin
        print('Testing terminated | Training time=%d s' % (end))

    def _preprocess_data_for_LSTM(self,X):
        """
        Preposses data and prepare sequence of data based on number of samples needed in a window
        Args:
            inputs : X , ndarray of size (number of sample,features)

        Returns:
            return : X , Y  X being samples till (t-1) of data and Y the t time data
        """
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:   # pragma: no cover
            X_norm = np.copy(X)

        X_data = []
        Y_data = []
        for index in range(X.shape[0] - self.window_size):
            X_data.append(X_norm[index:index+self.window_size])
            Y_data.append(X_norm[index+self.window_size])
        X_data = np.asarray(X_data)
        Y_data = np.asarray(Y_data)

        return X_data,Y_data



    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. .
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])

        X = check_array(X)
        #print("inside")
        #print(X.shape)
        #print(X[0])
        X_norm,Y_norm = self._preprocess_data_for_LSTM(X)
        pred_scores = np.zeros(X.shape)
        pred_scores[self.window_size:] = self.model_.predict(X_norm)
        Y_norm_for_decision_scores = np.zeros(X.shape)
        Y_norm_for_decision_scores[self.window_size:] = Y_norm
        return pairwise_distances_no_broadcast(Y_norm_for_decision_scores, pred_scores)






