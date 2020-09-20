# -*- coding: utf-8 -*-
"""Autoregressive model for multivariate time series outlier detection.
"""
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .CollectiveBase import CollectiveBaseDetector
from pyod.models.pca import PCA as PCA_PYOD

from .utility import get_sub_matrices


class PCA(CollectiveBaseDetector):
    """PCA-based outlier detection with both univariate and multivariate
    time series data. TS data will be first transformed to tabular format. 
    For univariate data, it will be in shape of [valid_length, window_size].
    for multivariate data with d sequences, it will be in the shape of 
    [valid_length, window_size].

    Parameters
    ----------
    window_size : int
        The moving window size.

    step_size : int, optional (default=1)
        The displacement for moving window.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_components : int, float, None or string
        Number of components to keep. It should be smaller than the window_size.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
        to guess the dimension
        if ``0 < n_components < 1`` and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components
        n_components cannot be equal to n_features for svd_solver == 'arpack'.

    n_selected_components : int, optional (default=None)
        Number of selected principal components
        for calculating the outlier scores. It is not necessarily equal to
        the total number of the principal components. If not set, use
        all principal components.

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

    weighted : bool, optional (default=True)
        If True, the eigenvalues are used in score computation.
        The eigenvectors with small eigenvalues comes with more importance
        in outlier score calculation.

    standardization : bool, optional (default=True)
        If True, perform standardization first to convert
        data to zero mean and unit variance.
        See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
        
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, window_size, step_size=1, contamination=0.1,
                 n_components=None, n_selected_components=None,
                 copy=True, whiten=False, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=None,
                 weighted=True, standardization=True):
        super(PCA, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.step_size = step_size

        # parameters for PCA
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted
        self.standardization = standardization

        # initialize a kNN model
        self.model_ = PCA_PYOD(n_components=self.n_components,
                               n_selected_components=self.n_selected_components,
                               contamination=self.contamination,
                               copy=self.copy,
                               whiten=self.whiten,
                               svd_solver=self.svd_solver,
                               tol=self.tol,
                               iterated_power=self.iterated_power,
                               random_state=self.random_state,
                               weighted=self.weighted,
                               standardization=self.standardization)

    def fit(self, X: np.array) -> object:
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X).astype(np.float)

        # first convert it into submatrices, and flatten it
        sub_matrices, self.left_inds_, self.right_inds_ = get_sub_matrices(
            X,
            self.window_size,
            self.step_size,
            return_numpy=True,
            flatten=True,
            flatten_order='F')

        # if self.n_components > sub_matrices.shape[1]:
        #     raise ValueError('n_components exceeds window_size times the number of sequences.')

        # fit the PCA model
        self.model_.fit(sub_matrices)
        self.decision_scores_ = self.model_.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, X: np.array):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_'])
        X = check_array(X).astype(np.float)
        # first convert it into submatrices, and flatten it
        sub_matrices, X_left_inds, X_right_inds = get_sub_matrices(
            X,
            self.window_size,
            self.step_size,
            return_numpy=True,
            flatten=True,
            flatten_order='F')

        # return the prediction result by PCA
        return self.model_.decision_function(
            sub_matrices), X_left_inds.ravel(), X_right_inds.ravel()


if __name__ == "__main__":
    # X_train = np.asarray(
    #     [3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]).reshape(-1, 1)

    # X_test = np.asarray(
    #     [3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]).reshape(-1,
    #                                                                        1)

    X_train = np.asarray(
        [[3., 5], [5., 9], [7., 2], [42., 20], [8., 12], [10., 12],
         [12., 12],
         [18., 16], [20., 7], [18., 10], [23., 12], [22., 15]])

    w = get_sub_matrices(X_train, window_size=3, step=2, flatten=False)
    X_test = np.asarray(
        [[12., 10], [8., 12], [80., 80], [92., 983],
         [18., 16], [20., 7], [18., 10], [3., 5], [5., 9], [23., 12],
         [22., 15]])

    clf = PCA(window_size=3, step_size=2, contamination=0.2)

    clf.fit(X_train)
    decision_scores, left_inds_, right_inds = clf.decision_scores_, \
                                              clf.left_inds_, clf.right_inds_
    print(clf.left_inds_, clf.right_inds_)
    pred_scores, X_left_inds, X_right_inds = clf.decision_function(X_test)
    pred_labels, X_left_inds, X_right_inds = clf.predict(X_test)
    pred_probs, X_left_inds, X_right_inds = clf.predict_proba(X_test)

    print(pred_scores)
    print(pred_labels)
    print(pred_probs)
