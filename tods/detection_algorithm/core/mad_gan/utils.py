#!/usr/bin/env ipython
# Utility functions that don't fit in other scripts
import argparse
import json

def rgan_options_parser():
    """
    Define parser to parse options from command line, with defaults.
    Refer to this function for definitions of various variables.
    """
    parser = argparse.ArgumentParser(description='Train a GAN to generate sequential, real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    # options pertaining to data
    parser.add_argument('--data', help='what kind of data to train with?',
            default='gp_rbf',
            choices=['gp_rbf', 'sine', 'mnist', 'load'])
    # parser.add_argument('--num_samples', type=int, help='how many training examples \
    #                 to generate?', default=28*5*100)
    # parser.add_argument('--num_samples_t', type=int, help='how many testing examples \
    #                     for anomaly detection?', default=28 * 5 * 100)
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--num_signals', type=int, default=1)
    parser.add_argument('--normalise', type=bool, default=False, help='normalise the \
            training/vali/test data (during split)?')
    # parser.add_argument('--AD', type=bool, default=False, help='should we conduct anomaly detection?')

    ### for gp_rbf
    parser.add_argument('--scale', type=float, default=0.1)
            ### for sin (should be using subparsers for this...)
    parser.add_argument('--freq_low', type=float, default=1.0)
    parser.add_argument('--freq_high', type=float, default=5.0)
    parser.add_argument('--amplitude_low', type=float, default=0.1)
    parser.add_argument('--amplitude_high', type=float, default=0.9)
            ### for mnist
    parser.add_argument('--multivariate_mnist', type=bool, default=False)
    parser.add_argument('--full_mnist', type=bool, default=False)
            ### for loading
    parser.add_argument('--data_load_from', type=str, default='')
            ### for eICU
    parser.add_argument('--resample_rate_in_min', type=int, default=15)
    # hyperparameters of the model
    parser.add_argument('--hidden_units_g', type=int, default=100)
    parser.add_argument('--hidden_units_d', type=int, default=100)
    parser.add_argument('--hidden_units_e', type=int, default=100)
    parser.add_argument('--kappa', type=float, help='weight between final output \
            and intermediate steps in discriminator cost (1 = all \
            intermediate', default=1)
    parser.add_argument('--latent_dim', type=int, default=5, help='dimensionality \
            of the latent/noise space')
    parser.add_argument('--weight', type=int, default=0.5, help='weight of score')
    parser.add_argument('--degree', type=int, default=1, help='norm degree')
    parser.add_argument('--batch_mean', type=bool, default=False, help='append the mean \
            of the batch to all variables for calculating discriminator loss')
    parser.add_argument('--learn_scale', type=bool, default=False, help='make the \
            "scale" parameter at the output of the generator learnable (else fixed \
            to 1')
    # options pertaining to training
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--D_rounds', type=int, default=5, help='number of rounds \
            of discriminator training')
    parser.add_argument('--G_rounds', type=int, default=1, help='number of rounds \
            of generator training')
    parser.add_argument('--E_rounds', type=int, default=1, help='number of rounds \
               of encoder training')
    # parser.add_argument('--use_time', type=bool, default=False, help='enforce \
    #         latent dimension 0 to correspond to time')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--eval_mul', type=bool, default=False)
    parser.add_argument('--eval_an', type=bool, default=False)
    parser.add_argument('--eval_single', type=bool, default=False)
    parser.add_argument('--wrong_labels', type=bool, default=False, help='augment \
            discriminator loss with real examples with wrong (~shuffled, sort of) labels')
    # options pertaining to evaluation and exploration
    parser.add_argument('--identifier', type=str, default='test', help='identifier \
            string for output files')
    parser.add_argument('--sub_id', type=str, default='test', help='identifier \
               string for load parameters')
    # options pertaining to differential privacy
    parser.add_argument('--dp', type=bool, default=False, help='train discriminator \
            with differentially private SGD?')
    parser.add_argument('--l2norm_bound', type=float, default=1e-5,
            help='bound on norm of individual gradients for DP training')
    parser.add_argument('--batches_per_lot', type=int, default=1,
            help='number of batches per lot (for DP)')
    parser.add_argument('--dp_sigma', type=float, default=1e-5,
            help='sigma for noise added (for DP)')

    return parser

def load_settings_from_file(settings):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './tods/detection_algorithm/core/mad_gan/experiments/settings/' + settings['settings_file'] + '.txt'
    #settings_path = 'experiments/settings/' + settings['settings_file'] + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'))
    # check for settings missing in file
    for key in settings.keys():
        if not key in settings_loaded:
            print(key, 'not found in loaded settings - adopting value from command line defaults: ', settings[key])
            # overwrite parsed/default settings with those read from file, allowing for
    # (potentially new) default settings not present in file
    settings.update(settings_loaded)
    return settings
