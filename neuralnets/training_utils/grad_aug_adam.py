import tensorflow as tf
from typing import List, Union


class GradAugAdam(tf.keras.optimizers.Adam):
    """
    Custom optimizer that adds Gaussian noise to gradients during optimization.
    Inherits from tf.keras.optimizers.Adam.
    """

    def __init__(self, noise_stddev: float = 0.01, **kwargs):
        """
        Initializes the GradAugAdam optimizer.

        Args:
            noise_stddev (float): Standard deviation of the Gaussian noise to be added to gradients.
            **kwargs: Additional arguments to be passed to the parent class constructor.
        """
        super(GradAugAdam, self).__init__(**kwargs)
        self.noise_stddev = noise_stddev

    def get_gradients(
        self, loss: Union[float, tf.Tensor], params: List[tf.Variable]
    ) -> List[tf.Tensor]:
        """
        Computes the gradients of the loss with respect to the parameters and adds Gaussian noise to the gradients.

        Args:
            loss (float or tf.Tensor): The loss value.
            params (List[tf.Variable]): The list of trainable parameters.

        Returns:
            List[tf.Tensor]: The gradients of the loss with respect to the parameters, with added Gaussian noise.
        """
        grads = super(GradAugAdam, self).get_gradients(loss, params)
        noise = tf.random.normal(
            shape=tf.shape(grads), mean=0.0, stddev=self.noise_stddev, dtype=grads.dtype
        )
        return [grad + noise for grad, noise in zip(grads, noise)]
