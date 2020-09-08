#!/usr/bin/env python3

import sys
import unittest

runner = unittest.TextTestRunner(verbosity=1)

tests = unittest.TestLoader().discover('tests')

if not runner.run(tests).wasSuccessful():
    sys.exit(1)
