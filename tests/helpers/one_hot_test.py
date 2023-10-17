import unittest
import numpy as np
from utils.helpers.one_hot_helper import OneHotHelper
from utils.logging_utils import Logger

logger = Logger(__name__)

class TestOneHotHelper(unittest.TestCase):
    def setUp(self):
        self.helper = OneHotHelper()

    def test_to_one_hot(self):
        # Test case 1: Normal case with multiple unique labels
        y = np.array([1, 2, 3, 2, 1])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        result = self.helper.to_one_hot(y)
        np.testing.assert_array_equal(result, expected)
        logger.debug("Test case 1 passed")

        # Test case 2: All elements are the same
        y = np.array([1, 1, 1, 1, 1])
        expected = np.array([[1], [1], [1], [1], [1]])
        result = self.helper.to_one_hot(y)
        np.testing.assert_array_equal(result, expected)
        logger.debug("Test case 2 passed")

        # Test case 3: Normal case with non-sequential labels
        y = np.array([1, 3, 5, 3, 1])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        result = self.helper.to_one_hot(y)
        np.testing.assert_array_equal(result, expected)
        logger.debug("Test case 3 passed")

        # Test case 4: Normal case with negative labels
        y = np.array([-1, -2, -3, -2, -1])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        result = self.helper.to_one_hot(y)
        np.testing.assert_array_equal(result, expected)
        logger.debug("Test case 4 passed")
