import unittest
import numpy as np
from modeltrainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.samples = 100
        self.X = np.random.rand(self.samples, 801, 7)
        self.y = np.zeros((self.samples, 5))
        self.y[np.arange(self.samples), np.random.randint(5, size=self.samples)] = 1
        self.modeltrainer = ModelTrainer()

    def test_k_fold_validation(self):
        # Test with a valid model name
        histories, models = self.modeltrainer.k_fold_validation(
            self.X, self.y, model_name="OneDCNN", k=2, epochs=1, electrodes=7
        )
        self.assertEqual(len(histories), 2)
        self.assertEqual(len(models), 2)

        # Test with an invalid model name
        with self.assertRaises(ValueError):
            self.modeltrainer.k_fold_validation(
                self.X, self.y, model_name="InvalidModel"
            )

    def test_benchmark_prediction(self):
        # Train a model first
        _, models = self.modeltrainer.k_fold_validation(
            self.X, self.y, model_name="OneDCNN", k=2, epochs=1, electrodes=7
        )
        model = models[0]

        predictions, elapsed_time_ms = self.modeltrainer.benchmark_prediction(
            model, self.X
        )
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(elapsed_time_ms, float)
