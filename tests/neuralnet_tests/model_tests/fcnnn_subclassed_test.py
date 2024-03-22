import numpy as np
import tensorflow as tf
import unittest
from neuralnets.models.fcnnmodel_subclassed import FCNNModel


class TestFCNNModel(unittest.TestCase):
    def setUp(self):
        self.load_level = 0
        self.electrodes = 7
        self.model = FCNNModel.create_and_compile(
            load_level=self.load_level, electrodes=self.electrodes
        )

        # Create some random training data
        self.x_train = np.random.rand(10, 641, self.electrodes)
        self.y_train = np.random.randint(0, 5, 10)

        # Convert labels to categorical
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 5)

    def test_create_and_compile(self):
        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertEqual(self.model._name, f"FCNN_ch_level_{self.load_level}")
        self.assertEqual(self.model.loss, tf.keras.losses.categorical_crossentropy)

        # Check the optimizer
        self.assertIsInstance(self.model.optimizer, tf.keras.optimizers.Optimizer)

        # Check the number of layers in the model
        self.assertGreater(len(self.model.layers), 0)

        # Check the type of the first layer
        self.assertIsInstance(self.model.layers[0], tf.keras.layers.Layer)

    def test_call(self):
        # Create a random tensor to pass to the call method
        input_tensor = tf.random.uniform((1, 641, self.electrodes, 1))
        output_tensor = self.model.call(input_tensor)

        # Check that the output is a tensor
        self.assertIsInstance(output_tensor, tf.Tensor)

        # Check that the output shape is as expected
        self.assertEqual(output_tensor.shape, (1, 5))

    def test_fit(self):
        # Fit the model, note batch size is 10 - the size of the training data
        history = self.model.fit(
            self.x_train, self.y_train, epochs=1, verbose=0, batch_size=10
        )

        # Check that history object is returned
        self.assertIsInstance(history, tf.keras.callbacks.History)

        # Check that history object has loss and accuracy
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
