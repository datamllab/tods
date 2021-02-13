import numpy as np
import pandas as pd
import pdb
import re
from time import time
import json
import random

from tods.detection_algorithm.core.mad_gan import model

from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal, invgamma, mode
from scipy.special import gamma
#from scipy.misc.pilutil import imresize
from functools import partial
from math import ceil

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


# --- deal with the SWaT data --- #
def swat(seq_length, seq_step, num_signals, randomize=False):
    """ Load and serialise """
    # train = np.load('./data/swat.npy')
    # print('Loaded swat from .npy')
    train = np.loadtxt(open('./data/swat.csv'), delimiter=',')
    print('Loaded swat from .csv')
    m, n = train.shape # m=496800, n=52
    for i in range(n - 1):
        A = max(train[:, i])
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[21600:, 0:n-1]
    labels = train[21600:, n-1]    # the last colummn is label
    #############################
    # -- choose variable for uni-variate GAN-AD -- #
    # samples = samples[:, [1, 8, 18, 28]]
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    from sklearn.decomposition import PCA
    # ALL SENSORS IDX
    # XS = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
    # X_n = samples[:, XS]
    # X_a = samples_a[:, XS]
    # All VARIABLES
    X_n = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=5 -- #
    n_components = num_signals
    pca = PCA(n_components, svd_solver='full')
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_

    # projected values on the principal component
    T_n = np.matmul(X_n, pc.transpose(1, 0))
    samples = T_n

    # # only for one-dimensional
    # samples = T_n.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    # seq_length = 7200
    num_samples = (samples.shape[0]-seq_length)//seq_step
    print("num_samples:", num_samples)
    print("num_signals:", num_signals)
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
       bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1,1])
       for i in range(num_signals):
           aa[j, :, i] = samples[(j * seq_step):(j*seq_step + seq_length), i]

    # samples = aa[:, 0:7200:200, :]
    # labels = bb[:, 0:7200:200, :]
    samples = aa
    labels = bb

    return samples, labels

def swat_birgan(seq_length, seq_step, num_signals, randomize=False):
    """ Load and serialise """
    # train = np.load('./data/swat.npy')
    # print('Loaded swat from .npy')
    train = np.loadtxt(open('./data/swat.csv'), delimiter=',')
    print('Loaded swat from .csv')
    m, n = train.shape # m=496800, n=52
    for i in range(n - 1):
        A = max(train[:, i])
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[21600:, 0:n-1]
    labels = train[21600:, n-1]    # the last colummn is label
    #############################
    # # -- choose variable for uni-variate GAN-AD -- #
    # # samples = samples[:, [1, 8, 18, 28]]
    ###########################################
    ###########################################
    nn = samples.shape[1]
    num_samples = (samples.shape[0]-seq_length)//seq_step
    aa = np.empty([num_samples, nn, nn])
    AA = np.empty([seq_length, nn])
    bb = np.empty([num_samples, seq_length, 1])

    print('Pre-process training data...')
    for j in range(num_samples):
       # display batch progress
       model_bigan.display_batch_progression(j, num_samples)
       bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1,1])
       for i in range(nn):
           AA[:, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
       aa[j, :, :] = np.cov(AA.T)

    samples = aa
    labels = bb

    return samples, labels

def swat_test(seq_length, seq_step, num_signals, randomize=False):
    """ Load and serialise """
    # test = np.load('./data/swat_a.npy')
    # print('Loaded swat_a from .npy')
    test = np.loadtxt(open('./data/swat_a.csv'), delimiter=',')
    print('Loaded swat_a from .csv')
    m, n = test.shape  # m1=449919, n1=52
    for i in range(n - 1):
        B = max(test[:, i])
        if B != 0:
            test[:, i] /= max(test[:, i])
            # scale from -1 to 1
            test[:, i] = 2 * test[:, i] - 1
        else:
            test[:, i] = test[:, i]

    samples = test[:, 0:n - 1]
    labels = test[:, n - 1]
    idx = np.asarray(list(range(0, m)))  # record the idx of each point
    #############################
    # -- choose variable for uni-variate GAN-AD -- #
    # samples = samples[:, [1,2,3,4]]
    # samples_a = samples_a[:, [1,2,3,4]]
    ############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    from sklearn.decomposition import PCA
    import DR_discriminator as dr
    # ALL SENSORS IDX
    # XS = [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47]
    # X_n = samples[:, XS]
    # X_a = samples_a[:, XS]
    # All VARIABLES
    X_a = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=5 -- #
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))

    samples = T_a
    # # only for one-dimensional
    # samples = T_a.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb

    return samples, labels, index


def swat_birgan_test(seq_length, seq_step, num_signals, randomize=False):
    """ Load and serialise """
    # train = np.load('./data/swat.npy')
    # print('Loaded swat from .npy')
    test = np.loadtxt(open('./data/swat_a.csv'), delimiter=',')
    print('Loaded swat_a from .csv')
    m, n = test.shape  # m1=449919, n1=52
    for i in range(n - 1):
        B = max(test[:, i])
        if B != 0:
            test[:, i] /= max(test[:, i])
            # scale from -1 to 1
            test[:, i] = 2 * test[:, i] - 1
        else:
            test[:, i] = test[:, i]

    samples = test[:, 0:n - 1]
    labels = test[:, n - 1]
    # idx = np.asarray(list(range(0, m)))  # record the idx of each point
    #############################
    # # -- choose variable for uni-variate GAN-AD -- #
    # # samples = samples[:, [1, 8, 18, 28]]
    ###########################################
    ###########################################
    nn = samples.shape[1]
    num_samples = (samples.shape[0]-seq_length)//seq_step
    aa = np.empty([num_samples, nn, nn])
    AA = np.empty([seq_length, nn])
    bb = np.empty([num_samples, seq_length, 1])

    print('Pre-process testing data...')
    for j in range(num_samples):
       # display batch progress
       model_bigan.display_batch_progression(j, num_samples)
       bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1,1])
       for i in range(nn):
           AA[:, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]
       aa[j, :, :] = np.cov(AA.T)

    samples = aa
    labels = bb

    return samples, labels


def wadi(seq_length, seq_step, num_signals, randomize=False):
    train = np.load('./data/wadi.npy')
    print('Loaded wadi from .npy')
    m, n = train.shape  # m=1048571, n=119
    for i in range(n-1):
        A = max(train[:, i])
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[259200:, 0:n-1]  # normal
    labels = train[259200:, n-1]
    #############################
    samples = samples[:, [0, 3, 6, 17]]
    # samples = samples[:, 0]
    ############################
    # # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    # from sklearn.decomposition import PCA
    # import DR_discriminator as dr
    # X_n = samples
    # ####################################
    # ###################################
    # # -- the best PC dimension is chosen pc=8 -- #
    # n_components = num_signals
    # pca = PCA(n_components, svd_solver='full')
    # pca.fit(X_n)
    # pc = pca.components_
    # # projected values on the principal component
    # T_n = np.matmul(X_n, pc.transpose(1, 0))
    #
    # samples = T_n
    # # # only for one-dimensional
    # # samples = T_n.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    seq_length = 10800
    num_samples = (samples.shape[0] - seq_length) // seq_step
    print("num_samples:", num_samples)
    print("num_signals:", num_signals)
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        # aa[j, :, :] = np.reshape(samples[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa[:, 0:10800:300, :]
    labels = bb[:, 0:10800:300, :]

    return samples, labels


def wadi_test(seq_length, seq_step, num_signals, randomize=False):
    test = np.load('./data/wadi_a.npy')
    print('Loaded wadi_a from .npy')
    m, n = test.shape  # m1=172801, n1=119

    for i in range(n - 1):
        B = max(test[:, i])
        if B != 0:
            test[:, i] /= max(test[:, i])
            # scale from -1 to 1
            test[:, i] = 2 * test[:, i] - 1
        else:
            test[:, i] = test[:, i]

    samples = test[:, 0:n - 1]
    labels = test[:, n - 1]
    idx = np.asarray(list(range(0, m)))  # record the idx of each point
    #############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    from sklearn.decomposition import PCA
    import DR_discriminator as dr
    X_a = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=8 -- #
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))

    samples = T_a
    # # only for one-dimensional
    # samples = T_a.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * 10):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb

    return samples, labels, index

def kdd99(seq_length, seq_step, num_signals):
    train = np.load('./tods/detection_algorithm/core/mad_gan/data/kdd99_train.npy')
    print('load kdd99_train from .npy')
    m, n = train.shape  # m=562387, n=35
    # normalization
    for i in range(n - 1):
        # print('i=', i)
        A = max(train[:, i])
        # print('A=', A)
        if A != 0:
            train[:, i] /= max(train[:, i])
            # scale from -1 to 1
            train[:, i] = 2 * train[:, i] - 1
        else:
            train[:, i] = train[:, i]

    samples = train[:, 0:n - 1]
    labels = train[:, n - 1]  # the last colummn is label
    #############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    from sklearn.decomposition import PCA
    X_n = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=6 -- #
    n_components = num_signals
    pca = PCA(n_components, svd_solver='full')
    pca.fit(X_n)
    ex_var = pca.explained_variance_ratio_
    pc = pca.components_
    # projected values on the principal component
    T_n = np.matmul(X_n, pc.transpose(1, 0))
    samples = T_n
    # # only for one-dimensional
    # samples = T_n.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    num_samples = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples, seq_length, num_signals])
    bb = np.empty([num_samples, seq_length, 1])

    for j in range(num_samples):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb

    return samples, labels

def kdd99_test(seq_length, seq_step, num_signals):
    test = np.load('./data/kdd99_test.npy')
    print('load kdd99_test from .npy')

    m, n = test.shape  # m1=494021, n1=35

    for i in range(n - 1):
        B = max(test[:, i])
        if B != 0:
            test[:, i] /= max(test[:, i])
            # scale from -1 to 1
            test[:, i] = 2 * test[:, i] - 1
        else:
            test[:, i] = test[:, i]

    samples = test[:, 0:n - 1]
    labels = test[:, n - 1]
    idx = np.asarray(list(range(0, m)))  # record the idx of each point
    #############################
    ############################
    # -- apply PCA dimension reduction for multi-variate GAN-AD -- #
    from sklearn.decomposition import PCA
    import DR_discriminator as dr
    X_a = samples
    ####################################
    ###################################
    # -- the best PC dimension is chosen pc=6 -- #
    n_components = num_signals
    pca_a = PCA(n_components, svd_solver='full')
    pca_a.fit(X_a)
    pc_a = pca_a.components_
    # projected values on the principal component
    T_a = np.matmul(X_a, pc_a.transpose(1, 0))
    samples = T_a
    # # only for one-dimensional
    # samples = T_a.reshape([samples.shape[0], ])
    ###########################################
    ###########################################
    num_samples_t = (samples.shape[0] - seq_length) // seq_step
    aa = np.empty([num_samples_t, seq_length, num_signals])
    bb = np.empty([num_samples_t, seq_length, 1])
    bbb = np.empty([num_samples_t, seq_length, 1])

    for j in range(num_samples_t):
        bb[j, :, :] = np.reshape(labels[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb[j, :, :] = np.reshape(idx[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(num_signals):
            aa[j, :, i] = samples[(j * seq_step):(j * seq_step + seq_length), i]

    samples = aa
    labels = bb
    index = bbb

    return samples, labels, index


# ############################ data pre-processing #################################
# --- to do with loading --- #
# --- to do with loading --- #
def get_samples_and_labels(settings):
    """
    Parse settings options to load or generate correct type of data,
    perform test/train split as necessary, and reform into 'samples' and 'labels'
    dictionaries.
    """
    if settings['data_load_from']:
        data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
        print('Loading data from', data_path)
        samples, pdf, labels = get_data('load', data_path)
        train, vali, test = samples['train'], samples['vali'], samples['test']
        train_labels, vali_labels, test_labels = labels['train'], labels['vali'], labels['test']
        del samples, labels
    else:
        # generate the data
        data_vars = ['num_samples', 'num_samples_t','seq_length', 'seq_step', 'num_signals', 'freq_low',
                'freq_high', 'amplitude_low', 'amplitude_high', 'scale', 'full_mnist']
        data_settings = dict((k, settings[k]) for k in data_vars if k in settings.keys())
        samples, pdf, labels = get_data(settings['data'], settings['seq_length'], settings['seq_step'], settings['num_signals'], settings['sub_id'])
        if 'multivariate_mnist' in settings and settings['multivariate_mnist']:
            seq_length = samples.shape[1]
            samples = samples.reshape(-1, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
        if 'normalise' in settings and settings['normalise']: # TODO this is a mess, fix
            print(settings['normalise'])
            norm = True
        else:
            norm = False
        if labels is None:
            train, vali, test = split(samples, [0.6, 0.2, 0.2], normalise=norm)
            train_labels, vali_labels, test_labels = None, None, None
        else:
            train, vali, test, labels_list = split(samples, [0.6, 0.2, 0.2], normalise=norm, labels=labels)
            train_labels, vali_labels, test_labels = labels_list

    labels = dict()
    labels['train'], labels['vali'], labels['test'] = train_labels, vali_labels, test_labels

    samples = dict()
    samples['train'], samples['vali'], samples['test'] = train, vali, test

    # futz around with labels
    # TODO refactor cause this is messy
    if 'one_hot' in settings and settings['one_hot'] and not settings['data_load_from']:
        if len(labels['train'].shape) == 1:
            # ASSUME labels go from 0 to max_val inclusive, find max-val
            max_val = int(np.max([labels['train'].max(), labels['test'].max(), labels['vali'].max()]))
            # now we have max_val + 1 dimensions
            print('Setting cond_dim to', max_val + 1, 'from', settings['cond_dim'])
            settings['cond_dim'] = max_val + 1
            print('Setting max_val to 1 from', settings['max_val'])
            settings['max_val'] = 1

            labels_oh = dict()
            for (k, v) in labels.items():
                A = np.zeros(shape=(len(v), settings['cond_dim']))
                A[np.arange(len(v)), (v).astype(int)] = 1
                labels_oh[k] = A
            labels = labels_oh
        else:
            assert settings['max_val'] == 1
            # this is already one-hot!

    if 'predict_labels' in settings and settings['predict_labels']:
        samples, labels = data_utils.make_predict_labels(samples, labels)
        print('Setting cond_dim to 0 from', settings['cond_dim'])
        settings['cond_dim'] = 0

    # update the settings dictionary to update erroneous settings
    # (mostly about the sequence length etc. - it gets set by the data!)
    settings['seq_length'] = samples['train'].shape[1]
    settings['num_samples'] = samples['train'].shape[0] + samples['vali'].shape[0] + samples['test'].shape[0]
    settings['num_signals'] = samples['train'].shape[2]

    return samples, pdf, labels


def get_data(data_type, seq_length, seq_step, num_signals, sub_id, eval_single, eval_an, data_options=None):
    """
    Helper/wrapper function to get the requested data.
    """
    print('data_type')
    labels = None
    index = None
    if data_type == 'load':
        data_dict = np.load(data_options).item()
        samples = data_dict['samples']
        pdf = data_dict['pdf']
        labels = data_dict['labels']
    elif data_type == 'swat':
        samples, labels = swat(seq_length, seq_step, num_signals)
    elif data_type == 'swat_test':
        samples, labels, index = swat_test(seq_length, seq_step, num_signals)
    elif data_type == 'kdd99':
        samples, labels = kdd99(seq_length, seq_step, num_signals)
    elif data_type == 'kdd99_test':
        samples, labels, index = kdd99_test(seq_length, seq_step, num_signals)
    elif data_type == 'wadi':
        samples, labels = wadi(seq_length, seq_step, num_signals)
    elif data_type == 'wadi_test':
        samples, labels, index = wadi_test(seq_length, seq_step, num_signals)
    else:
        raise ValueError(data_type)
    print('Generated/loaded', len(samples), 'samples from data-type', data_type)
    return samples, labels, index


def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels is None:
        return samples[start_pos:end_pos], None
    else:
        if type(labels) == tuple: # two sets of labels
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]



def split(samples, proportions, normalise=False, scale=False, labels=None, random_seed=None):
    """
    Return train/validation/test split.
    """
    if random_seed != None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    assert np.sum(proportions) == 1
    n_total = samples.shape[0]
    n_train = ceil(n_total * proportions[0])
    n_test = ceil(n_total * proportions[2])
    n_vali = n_total - (n_train + n_test)
    # permutation to shuffle the samples
    shuff = np.random.permutation(n_total)
    train_indices = shuff[:n_train]
    vali_indices = shuff[n_train:(n_train + n_vali)]
    test_indices = shuff[(n_train + n_vali):]
    # TODO when we want to scale we can just return the indices
    assert len(set(train_indices).intersection(vali_indices)) == 0
    assert len(set(train_indices).intersection(test_indices)) == 0
    assert len(set(vali_indices).intersection(test_indices)) == 0
    # split up the samples
    train = samples[train_indices]
    vali = samples[vali_indices]
    test = samples[test_indices]
    # apply the same normalisation scheme to all parts of the split
    if normalise:
        if scale: raise ValueError(normalise, scale)  # mutually exclusive
        train, vali, test = normalise_data(train, vali, test)
    elif scale:
        train, vali, test = scale_data(train, vali, test)
    if labels is None:
        return train, vali, test
    else:
        print('Splitting labels...')
        if type(labels) == np.ndarray:
            train_labels = labels[train_indices]
            vali_labels = labels[vali_indices]
            test_labels = labels[test_indices]
            labels_split = [train_labels, vali_labels, test_labels]
        elif type(labels) == dict:
            # more than one set of labels!  (weird case)
            labels_split = dict()
            for (label_name, label_set) in labels.items():
                train_labels = label_set[train_indices]
                vali_labels = label_set[vali_indices]
                test_labels = label_set[test_indices]
                labels_split[label_name] = [train_labels, vali_labels, test_labels]
        else:
            raise ValueError(type(labels))
        return train, vali, test, labels_split
