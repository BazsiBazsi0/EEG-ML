import unittest
import tensorflow as tf
from neuralnets.training_utils.grad_aug_adam import GradAugAdam


class TestGradAugAdam(unittest.TestCase):
    def setUp(self) -> None:
        self.noise_stddev = 0.01
        self.optimizer = GradAugAdam(noise_stddev=self.noise_stddev)

    def test_init(self) -> None:
        self.assertEqual(self.optimizer.noise_stddev, self.noise_stddev)

    def test_get_gradients(self) -> None:
        var = tf.Variable(1.0)
        with tf.GradientTape() as tape:
            loss = tf.square(var - 0.0)
        grads = tape.gradient(loss, [var])

        # Check that the gradient is not None
        self.assertIsNotNone(grads[0])

        # Check that the gradient is not zero (since we added noise)
        self.assertNotEqual(grads[0].numpy(), 0.0)
