import unittest
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402


class TestAvailability(unittest.TestCase):
    def test_tensorflow_import(self):
        try:
            _ = tf.__version__
            self.assertTrue(True)
        except AttributeError:
            self.assertTrue(False, "TensorFlow is not available")

    def test_gpu_availability(self):
        try:
            gpus = tf.config.list_physical_devices("GPU")
            self.assertTrue(len(gpus) > 0, "No GPU available")
        except AttributeError:
            self.assertTrue(False, "TensorFlow is not available to check for GPUs")
