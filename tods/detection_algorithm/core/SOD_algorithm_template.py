# -*- coding: utf-8 -*-
"""Autoregressive model for univariate time series outlier detection.
"""
import numpy as np
# from .CollectiveBase import CollectiveBaseDetector
from pyod.models.knn import KNN


class SODetector: # Add the class to inherit here
    """
    Template of Supervised Oulier Detector

    """

    def __init__(self):

        pass



    def _build_model(self):
        model_ = KNN()
        return model_

    def fit(self, X: np.array, y: np.array, **kwargs) -> object:
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples, n_features)
            The annotations of input samples.
            For supervised outlier detection:
                0: normal sample.
                1: anomaly.

            For semi-supervised outlier detection:
                0: normal sample.
                1: anomaly.
                -1: non-annotated.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.learner = self._build_model()
        self.learner.fit(X)
        # print("FIT Finished!")

        return self

    def predict(self, X: np.array) -> np.array:

        # print(X)
        # y = np.array([0] * (X.shape[0]-1) + [1]).astype(np.int)
        # y = np.zeros(X.shape[0]).astype(np.int)
        y = self.learner.predict(X)
        # print(y)
        return y.astype('int').ravel()


