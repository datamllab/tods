# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import unittest
# noinspection PyProtectedMember
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
from sklearn.utils.testing import assert_raises

from sklearn.utils.estimator_checks import check_estimator

from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.data import generate_data


class CollectiveCommonTest:
    def __init__(self,
                 model,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 roc_floor,
                 ):
        self.clf = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.roc_floor = roc_floor

        self.clf.fit(self.X_train)

        pass

    def test_detector(self):

        self.test_parameters()
        self.test_train_scores()
        self.test_train_inds()
        self.test_prediction_scores()
        self.test_prediction_proba()
        self.test_prediction_proba_linear()
        self.test_prediction_proba_unify()
        self.test_prediction_proba_parameter()
        # self.test_fit_predict()
        # self.test_fit_predict_score()
        self.test_prediction_labels()
        self.test_prediction_inds()
        # self.test_predict_rank()
        # self.test_predict_rank_normalized()
        self.tearDown()

    def test_parameters(self):
        assert (hasattr(self.clf, 'decision_scores_') and
                self.clf.decision_scores_ is not None)
        assert (hasattr(self.clf, 'labels_') and
                self.clf.labels_ is not None)
        assert (hasattr(self.clf, 'threshold_') and
                self.clf.threshold_ is not None)
        assert (hasattr(self.clf, 'left_inds_') and
                self.clf.left_inds_ is not None)
        assert (hasattr(self.clf, 'right_inds_') and
                self.clf.right_inds_ is not None)
        assert (hasattr(self.clf, '_mu') and
                self.clf._mu is not None)
        assert (hasattr(self.clf, '_sigma') and
                self.clf._sigma is not None)

    def test_train_scores(self):
        assert_equal(len(self.clf.decision_scores_), self.y_train.shape[0])

    def test_train_inds(self):
        inds_valid = self.clf.left_inds_ < self.clf.right_inds_
        assert_equal(self.clf.left_inds_.shape, self.clf.decision_scores_.shape)
        assert_equal(self.clf.right_inds_.shape, self.clf.decision_scores_.shape)
        assert_equal(all(inds_valid), True)

    def test_prediction_scores(self):
        pred_scores, _, _ = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.y_test.shape[0])

        # check performance
        assert_greater(roc_auc_score(self.y_test, pred_scores), self.roc_floor)

    def test_prediction_labels(self):
        pred_labels, _, _ = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_inds(self):
        _, left_inds, right_inds = self.clf.predict(self.X_test)
        inds_valid = left_inds < right_inds

        assert_equal(left_inds.shape, self.y_test.shape)
        assert_equal(right_inds.shape, self.y_test.shape)
        assert_equal(all(inds_valid), True)


    def test_prediction_proba(self):
        pred_proba, _, _ = self.clf.predict_proba(self.X_test)
        assert_greater_equal(pred_proba.min(), 0)
        assert_less_equal(pred_proba.max(), 1)

    def test_prediction_proba_linear(self):
        pred_proba, _, _ = self.clf.predict_proba(self.X_test, method='linear')
        assert_greater_equal(pred_proba.min(), 0)
        assert_less_equal(pred_proba.max(), 1)

    def test_prediction_proba_unify(self):
        pred_proba, _, _ = self.clf.predict_proba(self.X_test, method='unify')
        assert_greater_equal(pred_proba.min(), 0)
        assert_less_equal(pred_proba.max(), 1)

    def test_prediction_proba_parameter(self):
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_fit_predict(self):
        pred_labels, _, _ = self.clf.fit_predict(X=self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_predict_rank(self):
        pred_socres, _, _ = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=2)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        pred_socres, _, _ = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)

        # assert the order is reserved
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=2)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def tearDown(self):
        pass
