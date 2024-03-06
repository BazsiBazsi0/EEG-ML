import unittest
import tensorflow as tf
from neuralnets.models.fcnnmodel_functional import FCNNModel
from neuralnets.training_utils.GradAugAdam import GradAugAdam


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

        # Check the model's compile metrics
        self.assertTrue(
            any(
                metric
                for metric in model.compiled_metrics._metrics
                if metric == "accuracy"
            )
        )
