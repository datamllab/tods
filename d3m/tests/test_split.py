import os
import sys
import unittest

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

COMMON_PRIMITIVES_TESTS_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives', 'tests')
sys.path.insert(0, COMMON_PRIMITIVES_TESTS_DIR)

import test_train_score_split


# We just reuse existings tests. This allows us to test the high-level data splitting class.
class TrainScoreDatasetSplitPrimitiveTestCase(test_train_score_split.TrainScoreDatasetSplitPrimitiveTestCase):
    pass


if __name__ == '__main__':
    unittest.main()
