import unittest
import numpy as np
from utils.helpers.smote_helper import SmoteHelper


class TestSmoteHelper(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(100, 5)
        self.y = np.random.choice([0, 1], size=(100,))

    def test_smote_processor(self):
        """
        Test the smote_processor method of the SmoteHelper class.
        """
        x_resampled, y_resampled = SmoteHelper.smote_processor(self.x, self.y)

        # Check if the resampled data has more instances than the original data
        self.assertTrue(len(x_resampled) >= len(self.x))
        self.assertTrue(len(y_resampled) >= len(self.y))
