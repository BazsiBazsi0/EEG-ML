import unittest
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import numpy as np
from neuralnets.training_utils.one_cycle_sched import OneCycleScheduler


class TestOneCycleScheduler(unittest.TestCase):
    def setUp(self):
        self.model = models.Sequential()
        self.model.add(layers.Dense(10, activation="relu", input_shape=(10,)))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.scheduler = OneCycleScheduler(
            max_lr=0.01,
            steps_per_epoch=100,
            epochs=10,
            div_factor=100.0,
            pct_start=0.3,
            verbose=0,
            patience=10,
        )
        self.scheduler.set_model(self.model)

    def test_initialization(self):
        self.assertEqual(self.scheduler.max_lr, 0.01)
        self.assertEqual(self.scheduler.total_steps, 1000)
        self.assertEqual(self.scheduler.step, 0)
        self.assertEqual(self.scheduler.verbose, 0)
        self.assertEqual(self.scheduler.patience, 10)

    def test_on_train_batch_begin(self):
        self.scheduler.on_train_batch_begin(0)
        self.assertEqual(self.scheduler.step, 1)
        self.assertAlmostEqual(float(self.model.optimizer.lr), 0.0001, places=4)

        self.scheduler = OneCycleScheduler(
            max_lr=0.01,
            steps_per_epoch=100,
            epochs=10,
            div_factor=100.0,
            pct_start=0.3,
            verbose=0,
            patience=10,
        )
        self.scheduler.model = self.model

    def test_initialization(self):
        self.assertEqual(self.scheduler.max_lr, 0.01)
        self.assertEqual(self.scheduler.total_steps, 1000)
        self.assertEqual(self.scheduler.step, 0)
        self.assertEqual(self.scheduler.verbose, 0)
        self.assertEqual(self.scheduler.patience, 10)

    def test_on_train_batch_begin(self):
        self.scheduler.on_train_batch_begin(0)
        self.assertEqual(self.scheduler.step, 1)
        self.assertAlmostEqual(float(self.model.optimizer.lr), 0.0001, places=4)
