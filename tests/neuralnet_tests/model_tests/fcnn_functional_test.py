import unittest
import numpy as np
import tensorflow as tf
from neuralnets.models.fcnnmodel_functional import FCNNModel
from neuralnets.training_utils.grad_aug_adam import GradAugAdam


class TestFCNNModel(unittest.TestCase):
    def setUp(self) -> None:
        self.load_level = 0
        self.electrodes = 7

    def test_create_and_compile_functional(self) -> None:
        model = FCNNModel.create_and_compile_functional(
            self.load_level, self.electrodes
        )
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.name, f"FCNN_ch_level_{self.load_level}")
        self.assertEqual(model.loss, "categorical_crossentropy")

        # Check the optimizer
        self.assertIsInstance(model.optimizer, GradAugAdam)

        # Check the number of layers in the model
        self.assertGreater(len(model.layers), 0)

        # Check the type of the first layer
        self.assertIsInstance(model.layers[0], tf.keras.layers.InputLayer)

        # Create dummy data
        x_train = np.random.random((100, 641, self.electrodes, 1))
        y_train = np.random.randint(5, size=(100, 1))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)

        # Train the model for one epoch
        model.fit(x_train, y_train, epochs=1, verbose=0)

        # Check the model's compile metrics
        self.assertIn("accuracy", model.metrics_names)
