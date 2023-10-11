import unittest


class TestAvailability(unittest.TestCase):
    def test_tensorflow_import(self):
        try:
            import tensorflow as tf

            self.assertTrue(True)
        except ImportError:
            self.assertTrue(False, "TensorFlow is not available")

    def test_gpu_availability(self):
        try:
            from tensorflow.python.client import device_lib

            gpus = [
                x.name
                for x in device_lib.list_local_devices()
                if x.device_type == "GPU"
            ]
            self.assertTrue(len(gpus) > 0, "No GPU available")
        except ImportError:
            self.assertTrue(False, "TensorFlow is not available to check for GPUs")

    def test_tensorrt_import(self):
        try:
            import tensorrt as trt

            self.assertTrue(True)
        except ImportError:
            self.assertTrue(False, "TensorRT is not available")
