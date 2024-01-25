import tensorflow as tf


class OneDCNN(tf.keras.Model):
    def __init__(self):
        super(OneDCNN, self).__init__()
        self._name = "1DCNN"
        self.drop_rate = 0.5

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=20, activation="relu", padding="same"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=20, activation="relu", padding="same"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=6, activation="relu", padding="same"
        )
        self.conv4 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=6, activation="relu", padding="same"
        )

        # Batch normalization layers
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.batch_n_2 = tf.keras.layers.BatchNormalization()

        # Dropout layers
        self.spatial_drop1 = tf.keras.layers.SpatialDropout1D(0.5)
        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

        # Pooling layer
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)

        # Flatten layer
        self.flat = tf.keras.layers.Flatten()

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(296, activation="relu")
        self.dense2 = tf.keras.layers.Dense(148, activation="relu")
        self.dense3 = tf.keras.layers.Dense(74, activation="relu")

        # Output layer
        self.out = tf.keras.layers.Dense(5, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_n_1(x)
        x = self.conv2(x)
        x = self.batch_n_2(x)
        x = self.spatial_drop1(x)
        x = self.conv3(x)
        x = self.avg_pool1(x)
        x = self.conv4(x)
        x = self.spatial_drop_2(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.out(x)
        return x
