import unittest
import tensorflow as tf
from neuralnets.models.fcnnmodel_subclassed import FCNNModel


class TestFCNNModel(unittest.TestCase):
    def setUp(self):
        self.load_level = 0
        self.electrodes = 7

    def test_create_and_compile(self):
        model = FCNNModel.create_and_compile(self.load_level, self.electrodes)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.name, f"FCNN_ch_level_{self.load_level}")
        self.assertEqual(model.loss, tf.keras.losses.categorical_crossentropy)

        # Check the optimizer
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Optimizer)

        # Check the number of layers in the model
        self.assertGreater(len(model.layers), 0)

        # Check the type of the first layer
        self.assertIsInstance(model.layers[0], tf.keras.layers.Layer)
