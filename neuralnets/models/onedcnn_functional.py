import tensorflow as tf
from neuralnets.training_utils.grad_aug_adam import GradAugAdam
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    SpatialDropout1D,
    AvgPool1D,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Sequential


class OneDCNNModel(tf.keras.Model):
    """
    Functional API model for a 1D Convolutional Neural Network
    """

    def create_and_compile_functional(
        load_level: int, electrodes: int
    ) -> tf.keras.Model:
        drop_rate = 0.5

        # Input layer
        inputs = Input(shape=(641, electrodes))

        # Convolutional layers
        x = Conv1D(filters=32, kernel_size=20, activation="relu", padding="same")(
            inputs
        )
        x = BatchNormalization()(x)
        x = Conv1D(filters=32, kernel_size=20, activation="relu", padding="valid")(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(drop_rate)(x)
        x = Conv1D(filters=32, kernel_size=6, activation="relu", padding="valid")(x)
        x = AvgPool1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=6, activation="relu", padding="valid")(x)
        x = SpatialDropout1D(drop_rate)(x)

        # Flatten layer
        x = Flatten()(x)

        # Fully connected layers
        x = Dense(296, activation="relu")(x)
        x = Dense(148, activation="relu")(x)
        x = Dense(74, activation="relu")(x)

        # Output layer
        outputs = Dense(5, activation="softmax")(x)

        # Create model
        model = Model(
            inputs=inputs, outputs=outputs, name=f"OneDCNN_ch_level_{load_level}"
        )

        # Compile model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=GradAugAdam(learning_rate=0.0001, noise_stddev=0.01),
            metrics=["accuracy"],
        )

        return model

    def create_and_compile_sequential_tune(
        hp, load_level: int, electrodes: int
    ) -> tf.keras.Model:
        drop_rate = 0.5

        # Create a Sequential model
        model = Sequential(name=f"OneDCNN_ch_level_{load_level}")

        filters = hp.Int("filters", min_value=32, max_value=512, step=32)
        kernel_size = hp.Int("kernel_size", min_value=1, max_value=30, step=3)

        # Convolutional layers
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                input_shape=(641, electrodes),
            )
        )
        model.add(BatchNormalization())
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )
        )
        model.add(BatchNormalization())
        model.add(SpatialDropout1D(drop_rate))
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )
        )
        model.add(AvgPool1D(pool_size=2))
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )
        )
        model.add(SpatialDropout1D(drop_rate))

        # Flatten layer
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(296, activation="relu"))
        model.add(Dense(148, activation="relu"))
        model.add(Dense(74, activation="relu"))

        # Output layer
        model.add(Dense(5, activation="softmax"))

        # Compile model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=GradAugAdam(learning_rate=0.0001, noise_stddev=0.01),
            metrics=["accuracy"],
        )

        return model
