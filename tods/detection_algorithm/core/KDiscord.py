# -*- coding: utf-8 -*-
"""Autoregressive model for multivariate time series outlier detection.
"""
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .CollectiveBase import CollectiveBaseDetector
from pyod.models.knn import KNN

from .utility import get_sub_matrices


# TODO: add an argument to exclude "near equal" samples
# TODO: another thought is to treat each dimension independent
class KDiscord(CollectiveBaseDetector):
    """KDiscord first split multivariate time series into 
    subsequences (matrices), and it use kNN outlier detection based on PyOD.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.
    
    See :cite:`aggarwal2015outlier,zhao2020using` for details.

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

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.

    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}

        - 'largest': use the distance to the kth neighbor as the outlier score
        - 'mean': use the average of all k neighbors as the outlier score
        - 'median': use the median of the distance to k neighbors as the
          outlier score

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for `radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

        .. deprecated:: 0.74
           ``algorithm`` is deprecated in PyOD 0.7.4 and will not be
           possible in 0.7.6. It has to use BallTree for consistency.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.

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
                 n_neighbors=5, method='largest',
                 radius=1.0, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None, n_jobs=1,
                 **kwargs):
        super(KDiscord, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.step_size = step_size

        # parameters for kNN
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        # initialize a kNN model
        self.model_ = KNN(contamination=self.contamination,
                          n_neighbors=self.n_neighbors,
                          radius=self.radius,
                          algorithm=self.algorithm,
                          leaf_size=self.leaf_size,
                          metric=self.metric,
                          p=self.p,
                          metric_params=self.metric_params,
                          n_jobs=self.n_jobs,
                          **kwargs)

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
            flatten=True)

        # fit the kNN model
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
            flatten=True)

        # return the prediction result by kNN
        return self.model_.decision_function(sub_matrices), \
               X_left_inds.ravel(), X_right_inds.ravel()


if __name__ == "__main__":
    X_train = np.asarray(
        [3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78,
         100]).reshape(-1, 1)

    X_test = np.asarray(
        [3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]).reshape(-1,
                                                                           1)

    # X_train = np.asarray(
    #     [[3., 5], [5., 9], [7., 2], [42., 20], [8., 12], [10., 12],
    #      [12., 12],
    #      [18., 16], [20., 7], [18., 10], [23., 12], [22., 15]])
    #
    # X_test = np.asarray(
    #     [[12., 10], [8., 12], [80., 80], [92., 983],
    #      [18., 16], [20., 7], [18., 10], [3., 5], [5., 9], [23., 12],
    #      [22., 15]])

    clf = KDiscord(window_size=3, step_size=1, contamination=0.2,
                   n_neighbors=5)

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
