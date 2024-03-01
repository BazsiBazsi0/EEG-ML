import tensorflow as tf
from neuralnets.training_utils.GradAugAdam import GradAugAdam
from neuralnets.training_utils.OneCycleScheduler import OneCycleScheduler


class OneDCNNModel(tf.keras.Model):
    """
    A subclass of tf.keras.Model for a 1D Convolutional Neural Network.
    Same as OneDCNN_F.py but with the call method.
    """

    def __init__(self):
        super(OneDCNNModel, self).__init__()
        self._name = "OneDCNN"
        self.drop_rate = 0.5

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=20, activation="relu", padding="same"
        )
        self.batch_n_1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=20, activation="relu", padding="same"
        )
        self.batch_n_2 = tf.keras.layers.BatchNormalization()
        self.spatial_drop1 = tf.keras.layers.SpatialDropout1D(0.5)
        self.conv3 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=6, activation="relu", padding="same"
        )
        self.avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)
        self.conv4 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=6, activation="relu", padding="same"
        )
        self.spatial_drop_2 = tf.keras.layers.SpatialDropout1D(self.drop_rate)

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

    def save_model_summary(self, filepath):
        with open(filepath, "w") as f:
            # Redirect the output of model.summary() to the file
            tf.keras.Model.summary(
                self, line_length=120, print_fn=lambda x: f.write(x + "\n")
            )
