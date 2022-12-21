import numpy as np
import pandas as pd
import os
from tods.sk_interface.timeseries_processing.SimpleExponentialSmoothing_skinterface import SimpleExponentialSmoothingSKI

from pyod.utils.data import generate_data
import unittest
from sklearn.metrics import roc_auc_score

class SimpleExponentialSmoothingSKI_TestCase(unittest.TestCase):
    def setUp(self):

        self.n_train = 200
        self.n_test = 100
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
             n_train=self.n_train, n_test=self.n_test, n_features=5,
             contamination=0., random_state=42)

        self.transformer = SimpleExponentialSmoothingSKI()

    def test_produce(self):
        X_transform = self.transformer.produce(self.X_test)
        


if __name__ == '__main__':
    unittest.main()
