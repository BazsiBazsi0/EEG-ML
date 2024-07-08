import unittest
import numpy as np
from fileprocessor import FileProcessor


class TestFileProcessor(unittest.TestCase):
    def setUp(self):
        x = np.random.rand(100, 10, 10)
        y = np.random.randint(0, 2, (100,))
        self.fp = FileProcessor(x, y)

    def test_to_one_hot(self):
        result = self.fp.to_one_hot(self.fp.y)
        self.assertEqual(result.shape[1], len(np.unique(self.fp.y)))

    def test_smote_processor(self):
        x_scaled = np.random.rand(100, 10)
        y_one_hot = np.random.randint(0, 2, 100)
        x_smote, y_smote = self.fp.smote_processor(x_scaled, y_one_hot)
        self.assertTrue(x_smote.shape[0] >= x_scaled.shape[0])
        self.assertEqual(x_smote.shape[1], x_scaled.shape[1])
        self.assertTrue(y_smote.shape[0] >= y_one_hot.shape[0])

    def test_equalize_samples(self):
        x_no_smote = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        y_one_hot = self.fp.to_one_hot(y)
        x_equalized, y_equalized = self.fp.equalize_samples(x_no_smote, y_one_hot)
        self.assertTrue(x_equalized.shape[0] <= x_no_smote.shape[0])
        self.assertEqual(x_equalized.shape[1], x_no_smote.shape[1])
        self.assertTrue(y_equalized.shape[0] <= y_one_hot.shape[0])
        self.assertEqual(y_equalized.shape[1], y_one_hot.shape[1])

    def test_preprocessor(self):
        x_no_smote, y_one_hot, x_val, y_val = self.fp.preprocessor()

        # earlier version conatined checks for x_smote, y_smote now they are removed
        # Check if the number of samples in x_no_smote, x_smote, and x_val are less than or equal to the original number of samples
        self.assertTrue(x_no_smote.shape[0] <= self.fp.x.shape[0])
        # self.assertTrue(x_smote.shape[0] <= self.fp.x.shape[0])
        self.assertTrue(x_val.shape[0] <= self.fp.x.shape[0])

        # Check if the second and third dimensions remain the same
        self.assertEqual(x_no_smote.shape[1:], self.fp.x.shape[1:])
        # self.assertEqual(x_smote.shape[1:], self.fp.x.shape[1:])
        self.assertEqual(x_val.shape[1:], self.fp.x.shape[1:])

        # Check if the number of samples in y_one_hot, y_smote, and y_val are less than or equal to the original number of samples
        self.assertTrue(y_one_hot.shape[0] <= self.fp.y.shape[0])
        self.assertEqual(y_one_hot.shape[1], len(np.unique(self.fp.y)))
        # self.assertTrue(y_smote.shape[0] <= self.fp.y.shape[0])
        self.assertTrue(y_val.shape[0] <= self.fp.y.shape[0])
        self.assertEqual(y_val.shape[1], len(np.unique(self.fp.y)))

        # Check if x_smote and y_smote have the same number of samples
        # self.assertEqual(x_smote.shape[0], y_smote.shape[0])

    def test_remove_majority_class(self):
        x = np.random.rand(100, 10)
        y = np.zeros((100, 2))
        y[:50, 0] = 1  # Class 0 is the majority
        y[50:, 1] = 1  # Class 1 is the minority

        new_x, new_y = FileProcessor.remove_majority_class(x, y)

        # Check if the majority class has been removed
        self.assertEqual(new_y.shape[1], y.shape[1] - 1)

        # Check if the number of samples in new_x and new_y are equal to the number of samples in the minority class
        self.assertEqual(new_x.shape[0], np.sum(y[:, 1]))
        self.assertEqual(new_y.shape[0], np.sum(y[:, 1]))

        # Check if the second dimension of new_x is the same as x
        self.assertEqual(new_x.shape[1], x.shape[1])

        # Check if the values in new_y are all 1 (since the majority class has been removed)
        self.assertTrue(np.all(new_y == 1))
