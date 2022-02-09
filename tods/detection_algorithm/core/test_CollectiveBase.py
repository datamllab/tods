# -*- coding: utf-8 -*-
from __future__ import division # pragma: no cover
from __future__ import print_function # pragma: no cover

import os # pragma: no cover
import sys # pragma: no cover

import unittest # pragma: no cover
from sklearn.utils.testing import assert_equal # pragma: no cover
from sklearn.utils.testing import assert_raises # pragma: no cover

import numpy as np # pragma: no cover

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # pragma: no cover

from detection_algorithm.core.CollectiveBase import CollectiveBaseDetector # pragma: no cover
from pyod.utils.data import generate_data # pragma: no cover


# Check sklearn\tests\test_base
# A few test classes
# noinspection PyMissingConstructor,PyPep8Naming
class MyEstimator(CollectiveBaseDetector): # pragma: no cover

    def __init__(self, l1=0, empty=None): # pragma: no cover
        self.l1 = l1
        self.empty = empty

    def fit(self, X, y=None): # pragma: no cover
        pass

    def decision_function(self, X): # pragma: no cover
        pass


# noinspection PyMissingConstructor
class K(CollectiveBaseDetector): # pragma: no cover
    def __init__(self, c=None, d=None): # pragma: no cover
        self.c = c
        self.d = d

    def fit(self, X, y=None): # pragma: no cover
        pass

    def decision_function(self, X): # pragma: no cover
        pass


# noinspection PyMissingConstructor
class T(CollectiveBaseDetector): # pragma: no cover
    def __init__(self, a=None, b=None): # pragma: no cover
        self.a = a
        self.b = b

    def fit(self, X, y=None): # pragma: no cover
        pass

    def decision_function(self, X): # pragma: no cover
        pass


# noinspection PyMissingConstructor
class ModifyInitParams(CollectiveBaseDetector): # pragma: no cover
    """Deprecated behavior.
    Equal parameters but with a type cast.
    Doesn't fulfill a is a
    """

    def __init__(self, a=np.array([0])): # pragma: no cover
        self.a = a.copy()

    def fit(self, X, y=None): # pragma: no cover
        pass

    def decision_function(self, X): # pragma: no cover
        pass


# noinspection PyMissingConstructor
class VargEstimator(CollectiveBaseDetector): # pragma: no cover
    """scikit-learn estimators shouldn't have vargs."""

    def __init__(self, *vargs): # pragma: no cover
        pass

    def fit(self, X, y=None): # pragma: no cover
        pass

    def decision_function(self, X): # pragma: no cover
        pass


class Dummy1(CollectiveBaseDetector): # pragma: no cover
    def __init__(self, contamination=0.1): # pragma: no cover
        super(Dummy1, self).__init__(contamination=contamination)

    def decision_function(self, X): # pragma: no cover
        pass

    def fit(self, X, y=None): # pragma: no cover
        pass


class Dummy2(CollectiveBaseDetector): # pragma: no cover
    def __init__(self, contamination=0.1): # pragma: no cover
        super(Dummy2, self).__init__(contamination=contamination)

    def decision_function(self, X): # pragma: no cover
        pass

    def fit(self, X, y=None): # pragma: no cover
        return X


class Dummy3(CollectiveBaseDetector): # pragma: no cover
    def __init__(self, contamination=0.1): # pragma: no cover
        super(Dummy3, self).__init__(contamination=contamination)

    def decision_function(self, X): # pragma: no cover
        pass

    def fit(self, X, y=None): # pragma: no cover
        self.labels_ = X


class TestBASE(unittest.TestCase): # pragma: no cover
    def setUp(self): # pragma: no cover
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination)

    def test_init(self): # pragma: no cover
        """
        Test base class initialization

        :return:
        """
        self.dummy_clf = Dummy1()
        assert_equal(self.dummy_clf.contamination, 0.1)

        self.dummy_clf = Dummy1(contamination=0.2)
        assert_equal(self.dummy_clf.contamination, 0.2)

        with assert_raises(ValueError):
            Dummy1(contamination=0.51)

        with assert_raises(ValueError):
            Dummy1(contamination=0)

        with assert_raises(ValueError):
            Dummy1(contamination=-0.5)

    def test_fit(self): # pragma: no cover
        self.dummy_clf = Dummy2()
        assert_equal(self.dummy_clf.fit(0), 0)

    def test_fit_predict(self): # pragma: no cover
        # TODO: add more testcases

        self.dummy_clf = Dummy3()

        assert_equal(self.dummy_clf.fit_predict(0), 0)

    def test_predict_proba(self): # pragma: no cover
        # TODO: create uniform testcases
        pass

    def test_rank(self): # pragma: no cover
        # TODO: create uniform testcases
        pass

    def test_repr(self): # pragma: no cover
        # Smoke test the repr of the base estimator.
        my_estimator = MyEstimator()
        repr(my_estimator)
        test = T(K(), K())
        assert_equal(
            repr(test),
            "T(a=K(c=None, d=None), b=K(c=None, d=None))"
        )

        some_est = T(a=["long_params"] * 1000)
        assert_equal(len(repr(some_est)), 415)

    def test_str(self): # pragma: no cover
        # Smoke test the str of the base estimator
        my_estimator = MyEstimator()
        str(my_estimator)

    def test_get_params(self): # pragma: no cover
        test = T(K(), K())

        assert ('a__d' in test.get_params(deep=True))
        assert ('a__d' not in test.get_params(deep=False))

        test.set_params(a__d=2)
        assert (test.a.d == 2)
        assert_raises(ValueError, test.set_params, a__a=2)

    def tearDown(self): # pragma: no cover
        pass


if __name__ == '__main__': # pragma: no cover
    unittest.main()
