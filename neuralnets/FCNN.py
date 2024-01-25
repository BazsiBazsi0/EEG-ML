import tensorflow as tf
from typing import List


class FCNN(tf.keras.Model):
    def __init__(self, electrodes: int):
        """
        Initializes the FCNN model.

        Args:
            electrodes (int): Number of electrodes in the input data.
        """
        super(FCNN, self).__init__()
        self._name = "FCNN"
        self.drop_rate: float = 0.5

        self.convs: List[tf.keras.layers.Conv2D] = [
            (
                tf.keras.layers.Conv2D(32, (3, 3), padding="same")
                if i < 2
                else tf.keras.layers.Conv2D(64, (3, 3), padding="same")
            )
            for i in range(6)
        ]
        self.leaky_relus: List[tf.keras.layers.LeakyReLU] = [
            tf.keras.layers.LeakyReLU() for _ in range(8)
        ]
        self.dropouts: List[tf.keras.layers.Dropout] = [
            tf.keras.layers.Dropout(self.drop_rate) for _ in range(5)
        ]
        self.batch_norms: List[tf.keras.layers.BatchNormalization] = [
            tf.keras.layers.BatchNormalization() for _ in range(4)
        ]
        self.final_conv: tf.keras.layers.Conv2D = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), padding="same"
        )
        self.flatten: tf.keras.layers.Flatten = tf.keras.layers.Flatten()
        self.dense1: tf.keras.layers.Dense = tf.keras.layers.Dense(64)
        self.dense2: tf.keras.layers.Dense = tf.keras.layers.Dense(
            5, activation="softmax"
        )

    def call(self, inputs):
        """
        Forward pass of the FullyConvCNN model.

        Args:
            inputs: Input data to the model.

        Returns:
            Output of the model.
        """
        x = inputs
        for i in range(6):
            x = self.convs[i](x)
            x = self.leaky_relus[i](x)
            if i % 2 == 0:
                x = self.dropouts[i // 2](x)
            if i % 3 == 2:
                x = self.batch_norms[i // 3](x)
        x = self.final_conv(x)
        x = self.leaky_relus[6](x)
        x = self.dropouts[3](x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.leaky_relus[7](x)
        x = self.batch_norms[3](x)
        x = self.dropouts[4](x)
        x = self.dense2(x)
        return x
