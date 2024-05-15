from tensorflow.keras import Model
from neuralnets.training_utils.grad_aug_adam import GradAugAdam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    Dropout,
    BatchNormalization,
    Flatten,
    Dense,
)


class FCNNModel(Model):
    """
    Functional API model for a Fully Convolutional Neural Network
    """

    def create_and_compile_functional(load_level: int, electrodes: int) -> Model:
        # Define dropout rate
        drop_rate = 0.75

        # Define input shape
        inputs = Input(shape=(801, electrodes, 1))

        # Start of the model layers
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = LeakyReLU()(x)
        x = Dropout(drop_rate)(x)
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = Dropout(drop_rate)(x)
        x = Conv2D(64, (3, 3), padding="valid", strides=(2, 2))(x)
        x = LeakyReLU()(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_rate)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = Dropout(drop_rate)(x)
        x = Flatten()(x)
        x = Dense(64, kernel_regularizer=l1(0.001))(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(drop_rate)(x)

        # Define output layer
        outputs = Dense(5, activation="softmax")(x)

        # Create model
        model = Model(
            inputs=inputs, outputs=outputs, name=f"FCNN_ch_level_{load_level}"
        )

        # Compile model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=GradAugAdam(learning_rate=0.0001, noise_stddev=0.01),
            metrics=["accuracy"],
        )

        return model
