import numpy as np
import pandas as pd
import os
from tods.sk_interface.detection_algorithm.SOD_skinterface import SODSKI

from pyod.utils.data import generate_data
import unittest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from unittest import TestCase
from sklearn.metrics import roc_auc_score

class SODSKI_TestCase(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)

        self.transformer = SODSKI(contamination=self.contamination)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)


    if __name__ == '__main__':
        unittest.main()
    