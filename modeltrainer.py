import time
import numpy as np
from tensorflow.keras import backend as K  # type: ignore
import keras_tuner
import gc
from sklearn.model_selection import KFold, train_test_split
from typing import List
import modeltrainer
from neuralnets.models.fcnnmodel import FCNNModel
from neuralnets.models.onedcnn_functional import OneDCNNModel
from neuralnets.training_utils.OneCycleScheduler import OneCycleScheduler


class modeltrainer:
    # TODO: Testing and fix the ci/cd for the tests on github actions
    # https://theaisummer.com/unit-test-deep-learning/
    @staticmethod
    def k_fold_validation(
        X,
        y,
        k: int = 10,
        epochs: int = 50,
        model_name: str = "",
        load_level: int = 0,
        electrodes: int = 0,
        shuffle: bool = True,
    ):
        """
        Function to execute k-fold cross validation on the data with the picked model.
        Model: Either "FCNN" or "OneDCNN"
        """
        models: List[object] = []
        histories: List[object] = []

        kfold = KFold(n_splits=k, shuffle=True)
        # Define a dictionary that maps model names to classes
        model_classes = {
            "FCNN": FCNNModel,
            "OneDCNN": OneDCNNModel,
        }

        # Check if the model name is valid
        if model_name not in model_classes:
            raise ValueError(
                f"Invalid model name '{model_name}'. Valid names are {list(model_classes.keys())}."
            )

        for train_index, val_index in kfold.split(X):
            # Create and compile the model inside the fold, necessary to compile the model for each fold
            model = model_classes[model_name].create_and_compile_functional(
                load_level, electrodes
            )

            # Split the data into training and validation sets
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            batch_size = 32
            steps_per_epoch = len(X_train) // batch_size

            scheduler = OneCycleScheduler(
                max_lr=0.0001,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=1,
            )
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                verbose=1,
                callbacks=[scheduler],
            )
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            print(
                f"Fold: {len(models)}, val_loss: {val_loss}, val_accuracy: {val_accuracy}"
            )

            models.append(model)
            histories.append(history)
            # Clear the session and delete the model to free up memory(avoids OOM errors)
            K.clear_session()
            del model
            # must add garbage collection manually to avoid holding the mem of the deleted model
            gc.collect()

        return histories, models

    @staticmethod
    def benchmark(X, y, epochs=10, batch_size=32):
        # start the timer
        start_time = time.time()

        # train the model
        model, history, accuracy = modeltrainer.NeuralNets.train_model(
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
    def benchmark_prediction(model, X, batch_size=32):
        # start the timer
        start_time = time.time()

        # make predictions
        predictions = model.predict(X, batch_size=batch_size)

        # stop the timer
        end_time = time.time()

        # calculate the elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000

        # get the predicted classes
        predicted_classes = np.argmax(predictions, axis=1)

        # get the unique classes and their counts
        unique_classes, counts = np.unique(predicted_classes, return_counts=True)

        print(
            f"Time taken to make predictions on {len(X)} samples: {elapsed_time_ms} milliseconds"
        )

        for unique_class, count in zip(unique_classes, counts):
            print(f"Class {unique_class} is predicted {count} times")

        return predictions, elapsed_time_ms

    @staticmethod
    def tuner(
        X_train,
        y_train,
        X_val,
        y_val,
        load_level: int = 0,
        electrodes: int = 0,
    ):
        """
        A tuner method that uses the keras tuner to search for the best hyperparameters for the model.
        Example usage:
        nn.NeuralNets.tuner(
            x,
            y,
            x_val,
            y_val,
            load_level=load_level,
            electrodes=len(config.ch_level[load_level]),
        )
        """
        epochs = 25
        batch_size = 32
        scheduler = OneCycleScheduler(
            max_lr=0.0001,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            verbose=1,
        )
        tuner = keras_tuner.tuners.RandomSearch(
            lambda hp: OneDCNNModel.create_and_compile_sequential_tune(
                hp, load_level, electrodes
            ),
            objective="val_accuracy",
            max_trials=100,
            directory="my_dir",
            project_name="OneDCNN_F",
        )

        tuner.search_space_summary()

        tuner.search(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[scheduler],
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(best_hyperparameters.values)
