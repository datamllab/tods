#!/usr/bin/env python3

import sys
import unittest

runner = unittest.TextTestRunner(verbosity=1)
tests = unittest.TestLoader().discover('./')
if not runner.run(tests).wasSuccessful():
    sys.exit(1)

#for each in ['data_processing', 'timeseries_processing', 'feature_analysis', 'detection_algorithm']:
#    tests = unittest.TestLoader().discover(each)
#    if not runner.run(tests).wasSuccessful():
#        sys.exit(1)
