#this file generates the _main.py files to test the primitives
import re
import os
import sys
#sys.path.insert(0, 'tods/utils/skinterface')
print(sys.path)
with open('../utils/entry_points/entry_points_detection_algorithm.txt','r',encoding='utf-8') as f:
    entry_file = f.read()

output_dir = '../test/detection_algorithm'   #output directory


primitive_folder_start_loc_buf = [i.start()+2 for i in re.finditer('=', entry_file)]
primitive_start_loc_buf = [i.start()+1 for i in re.finditer(':', entry_file)]
primitive_end_loc_buf = [i.start() for i in re.finditer('\n', entry_file)]

for primitive_index, primitive_start_loc in enumerate(primitive_start_loc_buf):

    primitive_folder_start_loc = primitive_folder_start_loc_buf[primitive_index]
    primitive_end_loc = primitive_end_loc_buf[primitive_index]

    primitive_folder = entry_file[primitive_folder_start_loc:primitive_start_loc-1]
    primitive_name = entry_file[primitive_start_loc:primitive_end_loc]
    algorithm_name = primitive_name.replace('Primitive', '')

    # print(entry_file[primitive_folder_start_loc:primitive_start_loc-1])
    # print(entry_file[primitive_start_loc:primitive_end_loc])

    primitve_api_name = primitive_name.replace('Primitive', '_skinterface')
    class_name = primitive_name.replace('Primitive', 'SKI')

    import_line1 = """import numpy as np
import pandas as pd
import os
""" + 'from tods.sk_interface.detection_algorithm.'+ primitve_api_name + ' import ' + class_name + '\n\n'

    import_line2 = """from pyod.utils.data import generate_data
import unittest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from unittest import TestCase
from sklearn.metrics import roc_auc_score\n
"""

    if algorithm_name == "PCAODetector" or algorithm_name == "Telemanom" or algorithm_name == "KDiscordODetector":
        main_line1 = 'class ' + algorithm_name + 'SKI_TestCase(unittest.TestCase):\n' + \
    """    def setUp(self):

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
        self.window_size = 5
        self.l_s = 5
        self.n_predictions = 1
        self.X_train, self.X_test,self.y_train,  self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)\n
    """
        main_line3 = """    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)


    if __name__ == '__main__':
        unittest.main()
    """
        if algorithm_name == "Telemanom":
            main_line2 = '    self.transformer = ' + class_name + '(contamination=self.contamination, l_s=self.l_s, n_predictions=self.n_predictions)\n        self.y_train = self.y_train[self.l_s:-self.n_predictions]\n        self.y_test = self.y_test[self.l_s:-self.n_predictions]\n        self.transformer.fit(self.X_train)\n\n'

        else:
            main_line2 = '    self.transformer = ' + class_name + '(contamination=self.contamination, window_size=self.window_size)\n        \n        self.y_test = self.y_test[self.window_size-1:]\n        self.transformer.fit(self.X_train)\n\n'
    else:
        main_line1 = 'class ' + algorithm_name + 'SKI_TestCase(unittest.TestCase):\n' + \
    """    def setUp(self):

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
            contamination=self.contamination, random_state=42)\n
    """
        main_line2 = '    self.transformer = ' + class_name + '(contamination=self.contamination)\n        self.transformer.fit(self.X_train)\n\n'
        main_line3 = """    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)


    if __name__ == '__main__':
        unittest.main()
    """

    python_content = import_line1 + import_line2 + main_line1+main_line2+main_line3
    python_name = 'test_ski_' + algorithm_name + '.py'
    
    with open(os.path.join(output_dir, python_name), 'w', encoding='utf-8') as f:
        f.write(python_content)
    print(os.path.join(output_dir, python_name))
    print(python_content)


