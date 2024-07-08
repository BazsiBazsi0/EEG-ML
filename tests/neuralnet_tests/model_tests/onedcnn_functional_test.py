import unittest
import tensorflow as tf
from kerastuner.engine.hyperparameters import HyperParameters
from neuralnets.models.onedcnn_functional import OneDCNNModel


class TestOneDCNNModel(unittest.TestCase):
    def setUp(self):
        self.load_level = 0
        self.electrodes = 7

    def test_create_and_compile_functional(self):
        model = OneDCNNModel.create_and_compile_functional(
            self.load_level, self.electrodes
        )
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.name, f"OneDCNN_ch_level_{self.load_level}")
        self.assertEqual(model.loss, tf.keras.losses.categorical_crossentropy)

        # Build the model with an input shape
        model.build((None, 801, self.electrodes))

        # Check if the input shape of the model is (None, 641, electrodes)
        input_shape = model.layers[0]._batch_input_shape
        self.assertEqual(input_shape, (None, 801, self.electrodes))

    def test_create_and_compile_sequential_tune(self):
        hp = HyperParameters()
        hp.Fixed("filters", value=32)
        hp.Fixed("kernel_size", value=20)

        model = OneDCNNModel.create_and_compile_sequential_tune(
            hp, self.load_level, self.electrodes
        )
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.name, f"OneDCNN_ch_level_{self.load_level}")
        self.assertEqual(model.loss, tf.keras.losses.categorical_crossentropy)

        # Build the model with an input shape
        model.build((None, 801, self.electrodes))

        # Check the number and types of layers in the model
        self.assertGreater(len(model.layers), 0)
        self.assertIsInstance(model.layers[0], tf.keras.layers.Layer)
