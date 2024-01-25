import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import nn
from neuralnets.OneCycleScheduler import OneCycleScheduler
from neuralnets.FCNN import FCNN
from neuralnets.GradAugAdam import GradAugAdam
from neuralnets.EarlyStoppingLearningRateSpeedup import EarlyStoppingLearningRateSpeedup


class NeuralNets:
    @staticmethod
    def loo(X, y):
        acc = []
        models = []
        # perform LOO cross-validation
        num_batches = 10
        batch_size = X.shape[0] // num_batches

        for i in range(num_batches):
            # calculate the start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # create the training and validation sets
            X_train = np.concatenate([X[:start_idx], X[end_idx:]])
            y_train = np.concatenate([y[:start_idx], y[end_idx:]])
            X_train = X_train.reshape(
                (
                    int(X_train.shape[0] * X_train.shape[1]),
                    X_train.shape[2],
                    X_train.shape[3],
                )
            )
            y_train = y_train.reshape(
                (int(y_train.shape[0] * y_train.shape[1]), y_train.shape[2])
            )

            X_val = X[start_idx:end_idx]
            y_val = y[start_idx:end_idx]
            X_val = X_val.reshape(
                (int(X_val.shape[0] * X_val.shape[1]), X_val.shape[2], X_val.shape[3])
            )
            y_val = y_val.reshape(
                (int(y_val.shape[0] * y_val.shape[1]), y_val.shape[2])
            )

            # train and evaluate the model
            model = nn.NeuralNets.FullyConvCNN(electrodes=X.shape[2])
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0
            )
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"batch {i}: loss = {loss} accuracy = {accuracy}")

            acc.append(accuracy)
            models.append(model)
            nn.NeuralNets.save_accuracy_curves(
                history, 10, f"{model.name}_fold_{i}_level_0.png"
            )

        return history, models, acc, np.average(acc)

    @staticmethod
    def k_fold_validation(X, y, k=10):
        acc = []
        models = []
        kfold = KFold(n_splits=k, shuffle=True)

        for train_index, val_index in kfold.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # reshape the data
            X_train = X_train.reshape(
                (
                    int(X_train.shape[0] * X_train.shape[1]),
                    X_train.shape[2],
                    X_train.shape[3],
                )
            )
            y_train = y_train.reshape(
                (int(y_train.shape[0] * y_train.shape[1]), y_train.shape[2])
            )

            X_val = X_val.reshape(
                (int(X_val.shape[0] * X_val.shape[1]), X_val.shape[2], X_val.shape[3])
            )
            y_val = y_val.reshape(
                (int(y_val.shape[0] * y_val.shape[1]), y_val.shape[2])
            )

            # train and evaluate the model
            model = nn.NeuralNets.FullyConvCNN(electrodes=X.shape[2])
            # model = FCNN(electrodes=X.shape[2])
            model.compile(
                loss=tf.keras.losses.categorical_crossentropy,
                optimizer=GradAugAdam(learning_rate=0.01, noise_stddev=0.01),
                metrics=["accuracy"],
            )
            model._name = "FCNN"

            # TODO: docu
            # create a learning rate scheduler, pretty aggressive, avoid plateus
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=10,
                min_lr=0.000001,
            )
            early_stopping = EarlyStopping(monitor="val_loss", patience=20)
            early_stopping_speedup = EarlyStoppingLearningRateSpeedup(patience=10)

            batch_size = 32
            epochs = 50

            scheduler = OneCycleScheduler(
                max_lr=0.01,
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                verbose=1,
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[scheduler],
                verbose=1,
            )
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(f"Fold: {len(models)}, loss: {loss}, accuracy: {accuracy}")

            acc.append(accuracy)
            models.append(model)
            nn.NeuralNets.save_accuracy_curves(
                history, f"{model.name}_fold_{len(models)-1}_level_0.png"
            )

        return history, models, acc, np.average(acc)

    @staticmethod
    def train_model(X, y, epochs=100, batch_size=32):
        # split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        # reshape the data
        X_train = X_train.reshape(
            (
                int(X_train.shape[0] * X_train.shape[1]),
                X_train.shape[2],
                X_train.shape[3],
            )
        )
        y_train = y_train.reshape(
            (int(y_train.shape[0] * y_train.shape[1]), y_train.shape[2])
        )

        X_val = X_val.reshape(
            (int(X_val.shape[0] * X_val.shape[1]), X_val.shape[2], X_val.shape[3])
        )
        y_val = y_val.reshape((int(y_val.shape[0] * y_val.shape[1]), y_val.shape[2]))

        # train the model
        model = nn.NeuralNets.FullyConvCNN(electrodes=X.shape[2])
        # create a learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, min_lr=0.0001
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[reduce_lr],
            verbose=1,
        )
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

        nn.NeuralNets.save_accuracy_curves(history, f"{model.name}_level_0.png")

        return model, history, accuracy

    @staticmethod
    def benchmark(X, y, epochs=10, batch_size=32):
        # start the timer
        start_time = time.time()

        # train the model
        model, history, accuracy = nn.NeuralNets.train_model(
            X, y, epochs=epochs, batch_size=batch_size
        )

        # stop the timer
        end_time = time.time()

        # calculate the elapsed time
        elapsed_time = end_time - start_time

        print(
            f"Time taken to train the model for {epochs} epochs: {elapsed_time} seconds"
        )

        return elapsed_time

    @staticmethod
    def generator_processor(model, x, y):
        # Reshape the input data
        x = x.reshape((10, 41, 641, 160))

        # Create a generator function
        def generator():
            while True:
                yield x, y

        # Create a dataset using the generator function
        dataset = tf.data.Dataset.from_generator(
            generator,
            (tf.float32, tf.int32),
            (tf.TensorShape((None, 41, 641, 160)), tf.TensorShape((160, 5))),
        )
        # Create the model using the Keras functional API
        inputs = tf.keras.Input(shape=(41, 641, 160))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Fit the model using the generator function
        model.fit(dataset, epochs=5)

    @staticmethod
    def leave_one_out(x, y):
        models = {
            "1D-CNN": NeuralNets.one_d_cnn_multi(),
            # 'FCN': NeuralNets.FullyConvCNN()
        }
        # Sumarize the model
        for model in models:
            models[model].summary(line_length=100)

        history_loo = []
        acc_loop = []
        acc = []
        avg_acc = []

        def data_generator(x, y):
            for i in range(0, len(x)):
                # Leaving out one person from the training
                x_loop = np.append(x[:i], x[i + 1 :], 0)
                y_loop = np.append(y[:i], y[i + 1 :], 0)

                # Validation data: the left out person from train data
                x_val = x[i]
                y_val = y[i]

                yield x_loop, y_loop, x_val, y_val

        for model_name, model in models.items():
            for x_loop, y_loop, x_val, y_val in data_generator(x, y):
                print(
                    "Current model: ",
                    model_name,
                    " Current iteration: ",
                    len(history_loo) + 1,
                    "/",
                    len(x) * len(models),
                )

                if model_name != "FCN":
                    # Reshaping into 2D array
                    x_loop = x_loop.reshape(
                        (
                            int(x_loop.shape[0] * x_loop.shape[1]),
                            x_loop.shape[2],
                            x_loop.shape[3],
                        )
                    )
                    y_loop = y_loop.reshape(
                        (int(y_loop.shape[0] * y_loop.shape[1]), y_loop.shape[2])
                    )
                elif model_name == "FCN":
                    x_loop = x_loop.reshape(
                        (x_loop.shape[0] * x_loop.shape[1], x_loop.shape[2], 641, 1)
                    )
                    y_loop = y_loop.reshape((y_loop.shape[0] * y_loop.shape[1], 5))

                # Validation is done on person left out on from the training from the train set, silenced the output
                history = model.fit(
                    x_loop,
                    y_loop,
                    epochs=1,
                    batch_size=10,
                    validation_data=(x_val, y_val),
                    verbose=0,
                )
                history_loo.append(history)

                # Saving and averaging the accuracies, evaluation is done on the original dataset
                test_loss, test_acc = models[model_name].evaluate(x_val, y_val)

                acc_loop.append(test_acc)
                avg_acc_loop = np.average(acc_loop)

            acc.append(acc_loop)
            avg_acc.append(avg_acc_loop)
            acc_loop = []

        return models, history_loo, acc, avg_acc

    @staticmethod
    def one_d_cnn_multi():
        loss = tf.keras.losses.categorical_crossentropy
        optimizer = tf.keras.optimizers.Adam()

        drop_rate = 0.5

        inputs = tf.keras.Input(shape=(64, 641))
        conv1 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=20, activation="relu", padding="same"
        )(inputs)
        batch_n_1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=20, activation="relu", padding="same"
        )(batch_n_1)
        batch_n_2 = tf.keras.layers.BatchNormalization()(conv2)
        spatial_drop1 = tf.keras.layers.SpatialDropout1D(0.5)(batch_n_2)
        conv3 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=6, activation="relu", padding="same"
        )(spatial_drop1)
        avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)(conv3)
        conv4 = tf.keras.layers.Conv1D(
            filters=32, kernel_size=6, activation="relu", padding="same"
        )(avg_pool1)
        spatial_drop_2 = tf.keras.layers.SpatialDropout1D(drop_rate)(conv4)
        flat = tf.keras.layers.Flatten()(spatial_drop_2)
        dense1 = tf.keras.layers.Dense(296, activation="relu")(flat)
        dropout1 = tf.keras.layers.Dropout(drop_rate)(dense1)
        dense2 = tf.keras.layers.Dense(148, activation="relu")(dropout1)
        dropout2 = tf.keras.layers.Dropout(drop_rate)(dense2)
        dense3 = tf.keras.layers.Dense(74, activation="relu")(dropout2)
        dropout3 = tf.keras.layers.Dropout(drop_rate)(dense3)
        out = tf.keras.layers.Dense(5, activation="softmax")(dropout3)

        model = tf.keras.Model(inputs, out)
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        model._name = "1DCNN"
        return model

    @staticmethod
    def FullyConvCNN(electrodes: int):
        drop_rate = 0.5
        inputs = tf.keras.Input(shape=(electrodes, 641, 1))

        # First Convolutional Layer
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        # Second Convolutional Layer
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Third Convolutional Layer
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        # First pooling layer
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="valid", strides=(2, 2))(x)
        x = tf.keras.layers.LeakyReLU()(x)
        # Fourth Convolutional Layer
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)
        # Fifth Convolutional Layer
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Sixth Convolutional Layer
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(drop_rate)(x)

        # Flatten layer
        x = tf.keras.layers.Flatten()(x)

        # First Fully Connected Layer
        dense1 = tf.keras.layers.Dense(64)(x)
        dense1 = tf.keras.layers.LeakyReLU()(dense1)
        batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
        dropout1 = tf.keras.layers.Dropout(drop_rate)(batch_norm1)

        # Second Fully Connected Layer
        dense2 = tf.keras.layers.Dense(5, activation="softmax")(dropout1)
        out = dense2

        model = tf.keras.Model(inputs, out)
        return model

    # FFT processor which takes data before smote, applies FFT on it, then SMOTE on the frequency domain, then applies IFFT
    @staticmethod
    def fft_processor(x_train, y_train):
        x_train_shape = x_train.shape
        # Apply Fast Fourier Transform to the training data
        x_train_fft = np.fft.fft(x_train)

        y_2d = y_train.reshape((-1, y_train.shape[2]))

        magnitude = np.abs(x_train_fft)
        phase = np.angle(x_train_fft)
        magnitude_2D = magnitude.reshape(-1, magnitude.shape[2] * magnitude.shape[3])
        phase_2D = phase.reshape(-1, phase.shape[2] * phase.shape[3])

        # Apply SMOTE to the magnitude and phase of the FFT
        # Reshape the data to be suitable for SMOTE
        # x_2d = x_train_fft.reshape((-1, x_train_fft.shape[2] * x_train_fft.shape[3]))
        # y_2d = y_train.reshape((-1, y_train.shape[2]))

        # Use SMOTE to oversample the minority class
        sm = SMOTE(random_state=42)
        magnitude_2D, y = sm.fit_resample(magnitude_2D, y_2d)
        phase_2D, y = sm.fit_resample(phase_2D, y_2d)

        mag = magnitude_2D.ravel()
        phas = phase_2D.ravel()
        x = []
        for i in range(len(mag)):
            x.append(mag[i] * np.exp(1j * phas[i]))

        x = np.fft.ifft(x)
        magnitude = magnitude_2D.reshape(10, -1, x_train.shape[2], 641)
        phase = phase_2D.reshape(10, -1, x_train.shape[2], 641)
        y = y.reshape(10, -1, 5)

        # Return the processed data
        return x, y

    @staticmethod
    def metrics_generation(models, x_test, y_test, file_name):
        for i in range(len(models)):
            x_2d = x_test.reshape(
                (
                    int(x_test.shape[0] * x_test.shape[1]),
                    x_test.shape[2],
                    x_test.shape[3],
                )
            )
            y_predict = models[i].predict(x_2d)

            # convert from a one hot encode in string
            y_2d = y_test.reshape(
                (int(y_test.shape[0] * y_test.shape[1]), y_test.shape[2])
            )

            y_test_class = tf.argmax(y_2d, axis=1)
            y_predicted_classes = tf.argmax(y_predict, 1)

            with open(
                file_name + "_" + str(i) + "_classification_report.txt", "a"
            ) as file:
                file.write("Classification report: ")
                file.write(
                    classification_report(
                        y_test_class,
                        y_predicted_classes,
                        target_names=["left", "right", "fists", "rest", "feet"],
                    )
                )

                file.write("Confusion matrix: ")
                file.write(str(confusion_matrix(y_test_class, y_predicted_classes)))

    @staticmethod
    def metrics_csv(file_name, acc, average_accuracy):
        np.savetxt(
            file_name,
            np.column_stack([acc[0], acc[1]]),
            delimiter=",",
            fmt="%.4f",
            header="1DCNN,FCN",
            comments="",
        )
        with open(file_name, "a") as file:
            file.write(
                "\n"
                + str(np.round(average_accuracy[0], 4))
                + ","
                + str(np.round(average_accuracy[1], 4))
            )

    @staticmethod
    def save_accuracy_curves(history, filename):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs_range = range(len(acc))

        fig = plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="train accuracy")
        plt.plot(epochs_range, val_acc, label="validation accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="train loss")
        plt.plot(epochs_range, val_loss, label="validation loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        fig.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def plot_roc_curve(models, x_test, y_test, filename):
        for t in range(len(models)):
            plt.clf()
            labels = ["Left", "Right", "Fists", "Rest", "Feet"]
            x = x_test.reshape(
                x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3]
            )
            y = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2])
            y_pred = models[t].predict(x)
            auc = []
            with open(filename + ".txt", "a") as file:
                file.write("For model {}:\n".format(t))
            for i in range(5):
                fpr, tpr, _ = roc_curve(y[:, i], y_pred[:, i])
                plt.plot(fpr, tpr, label=labels[i])
                plt.legend(loc="lower right")
                auc.append(round(roc_auc_score(y[:, i], y_pred[:, i]), 4))
                with open(filename + ".txt", "a") as file:
                    file.write("AUC for class {} is {}\n".format(t, auc[i]))

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="upper right")
            plt.title("ROC, AUC=" + str(auc))
            plt.savefig(filename + "_" + str(t) + ".png")
