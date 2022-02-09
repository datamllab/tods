# -*- coding: utf-8 -*-
"""Utility functions for supporting time-series based outlier detection.
"""

import numpy as np
from sklearn.utils import check_array


# def get_sub_sequences(X, window_size, step=1):
#     """Chop a univariate time series into sub sequences.

#     Parameters
#     ----------
#     X : numpy array of shape (n_samples,)
#         The input samples.

#     window_size : int
#         The moving window size.

#     step_size : int, optional (default=1)
#         The displacement for moving window.

#     Returns
#     -------
#     X_sub : numpy array of shape (valid_len, window_size)
#         The numpy matrix with each row stands for a subsequence.
#     """
#     X = check_array(X).astype(np.float)
#     n_samples = len(X)

#     # get the valid length
#     valid_len = get_sub_sequences_length(n_samples, window_size, step)

#     X_sub = np.zeros([valid_len, window_size])
#     # y_sub = np.zeros([valid_len, 1])

#     # exclude the edge
#     steps = list(range(0, n_samples, step))
#     steps = steps[:valid_len]

#     for idx, i in enumerate(steps):
#         X_sub[idx,] = X[i: i + window_size].ravel()

#     return X_sub

def get_sub_matrices(X, window_size, step=1, return_numpy=True, flatten=True,
                     flatten_order='F'):
    """Chop a multivariate time series into sub sequences (matrices).

    Parameters
    ----------
    X : numpy array of shape (n_samples,)
        The input samples.

    window_size : int
        The moving window size.

    step_size : int, optional (default=1)
        The displacement for moving window.
    
    return_numpy : bool, optional (default=True)
        If True, return the data format in 3d numpy array.

    flatten : bool, optional (default=True)
        If True, flatten the returned array in 2d.
        
    flatten_order : str, optional (default='F')
        Decide the order of the flatten for multivarite sequences.
        ‘C’ means to flatten in row-major (C-style) order. 
        ‘F’ means to flatten in column-major (Fortran- style) order. 
        ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory, 
        row-major order otherwise. ‘K’ means to flatten a in the order the elements occur in memory. 
        The default is ‘F’.

    Returns
    -------
    X_sub : numpy array of shape (valid_len, window_size*n_sequences)
        The numpy matrix with each row stands for a flattend submatrix.
    """
    X = check_array(X).astype(np.float)
    n_samples, n_sequences = X.shape[0], X.shape[1]

    # get the valid length
    valid_len = get_sub_sequences_length(n_samples, window_size, step)

    X_sub = []
    X_left_inds = []
    X_right_inds = []

    # exclude the edge
    steps = list(range(0, n_samples, step))
    steps = steps[:valid_len]

    # print(n_samples, n_sequences)
    for idx, i in enumerate(steps):
        X_sub.append(X[i: i + window_size, :])
        X_left_inds.append(i)
        X_right_inds.append(i + window_size)

    X_sub = np.asarray(X_sub)

    if return_numpy:
        if flatten:
            temp_array = np.zeros([valid_len, window_size * n_sequences])
            if flatten_order == 'C':
                for i in range(valid_len):
                    temp_array[i, :] = X_sub[i, :, :].flatten(order='C')

            else:
                for i in range(valid_len):
                    temp_array[i, :] = X_sub[i, :, :].flatten(order='F')
            return temp_array, np.asarray(X_left_inds), np.asarray(
                X_right_inds)

        else:
            return np.asarray(X_sub), np.asarray(X_left_inds), np.asarray(
                X_right_inds)
    else:
        return X_sub, np.asarray(X_left_inds), np.asarray(X_right_inds)


def get_sub_sequences_length(n_samples, window_size, step):
    """Pseudo chop a univariate time series into sub sequences. Return valid
    length only.

    Parameters
    ----------
    X : numpy array of shape (n_samples,)
        The input samples.

    window_size : int
        The moving window size.

    step_size : int, optional (default=1)
        The displacement for moving window.

    Returns
    -------
    valid_len : int
        The number of subsequences.
        
    """
    # if X.shape[0] == 1:
    #     n_samples = X.shape[1]
    # elif X.shape[1] == 1:
    #     n_samples = X.shape[0]
    # else:
    #     raise ValueError("X is not a univarite series. The shape is {shape}.".format(shape=X.shape))

    # valid_len = n_samples - window_size + 1
    # valida_len = int_down(n_samples-window_size)/step + 1 
    valid_len = int(np.floor((n_samples - window_size) / step)) + 1
    return valid_len


if __name__ == "__main__":
    X_train = np.asarray(
        [3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78,
         100]).reshape(-1, 1)

    X_train = np.asarray(
        [[3., 5], [5., 9], [7., 2], [42., 20], [8., 12], [10., 12], [12., 12],
         [18., 16], [20., 7], [18., 10], [23., 12], [22., 15]])

    # n_samples = X.shape[0]

    window_size = 3

    # valid_len = n_samples - window_size + 1

    # X_sub = np.zeros([valid_len, window_size])

    # for i in range(valid_len):
    #     X_sub[i, ] = X[i: i+window_size]

    # X_sub_2 = get_sub_sequences(X, window_size, step=2)
    X_sub_3, X_left_inds, X_right_inds = get_sub_matrices(X_train, window_size,
                                                          step=2,
                                                          flatten_order='C')
