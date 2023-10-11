import unittest
import tensorflow as tf

class TestAvailability(unittest.TestCase):
    def test_tensorflow_import(self):
        try:
            _ = tf.__version__
            self.assertTrue(True)
        except AttributeError:
            self.assertTrue(False, "TensorFlow is not available")

    def test_gpu_availability(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            self.assertTrue(len(gpus) > 0, "No GPU available")
        except AttributeError:
            self.assertTrue(False, "TensorFlow is not available to check for GPUs")