import sys
import unittest

runner = unittest.TextTestRunner(verbosity=1)
tests = unittest.TestLoader().discover('./')
if not runner.run(tests).wasSuccessful():
    sys.exit(1)