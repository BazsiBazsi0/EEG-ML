import tensorflow as tf
from neuralnets.training_utils.grad_aug_adam import GradAugAdam
from neuralnets.training_utils.one_cycle_sched import OneCycleScheduler


class FCNNModel(tf.keras.Model):
    def __init__(self):
        """
        Fully Convolutional Neural Network Model.

        Args:
            electrodes (int): Number of electrodes.
        """
        super(FCNNModel, self).__init__()
        self._name = "FCNN"
        self.drop_rate = 0.5
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")
        self.leaky_relu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(self.drop_rate)
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")
        self.leaky_relu2 = tf.keras.layers.LeakyReLU()
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")
        self.leaky_relu3 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(self.drop_rate)
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), padding="valid", strides=(2, 2))
        self.leaky_relu4 = tf.keras.layers.LeakyReLU()
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")
        self.leaky_relu5 = tf.keras.layers.LeakyReLU()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(self.drop_rate)
        self.conv6 = tf.keras.layers.Conv2D(64, (3, 3), padding="same")
        self.leaky_relu6 = tf.keras.layers.LeakyReLU()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv7 = tf.keras.layers.Conv2D(128, (3, 3), padding="same")
        self.leaky_relu7 = tf.keras.layers.LeakyReLU()
        self.dropout4 = tf.keras.layers.Dropout(self.drop_rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64)
        self.leaky_relu8 = tf.keras.layers.LeakyReLU()
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.dropout5 = tf.keras.layers.Dropout(self.drop_rate)
        self.dense2 = tf.keras.layers.Dense(5, activation="softmax")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        x = tf.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.batch_norm1(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.conv5(x)
        x = self.leaky_relu5(x)
        x = self.batch_norm2(x)
        x = self.dropout3(x)
        x = self.conv6(x)
        x = self.leaky_relu6(x)
        x = self.batch_norm3(x)
        x = self.conv7(x)
        x = self.leaky_relu7(x)
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.leaky_relu8(x)
        x = self.batch_norm4(x)
        x = self.dropout5(x)
        x = self.dense2(x)
        return x

    @classmethod
    def create_and_compile(cls, load_level: int, electrodes: int) -> tf.keras.Model:
        model: tf.keras.Model = cls()
        model._name = f"{model._name}_ch_level_{load_level}"
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=GradAugAdam(learning_rate=0.01, noise_stddev=0.01),
            metrics=["accuracy"],
        )
        return model

    def fit(
        self,
        x=None,
        y=None,
        batch_size=32,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):

        # Calculate steps_per_epoch
        steps_per_epoch = len(x) // batch_size

        # Adding custom schedulers
        if callbacks is None:
            callbacks = []
        scheduler = OneCycleScheduler(
            max_lr=0.0001,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
        )
        callbacks.append(scheduler)

        # Call the parent fit method
        super().fit(
            x,
            y,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            shuffle,
            class_weight,
            sample_weight,
            initial_epoch,
            steps_per_epoch,
            validation_steps,
            validation_batch_size,
            validation_freq,
            max_queue_size,
            workers,
            use_multiprocessing,
        )

        return self.history
