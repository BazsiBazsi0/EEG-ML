import time
import gc
import numpy as np
from tensorflow.keras import backend as K  # type: ignore
from sklearn.model_selection import KFold
from neuralnets.models.fcnnmodel_functional import FCNNModel
from neuralnets.models.onedcnn_functional import OneDCNNModel
from neuralnets.training_utils.one_cycle_sched import OneCycleScheduler
from typing import List


class ModelTrainer:
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
        electrodes: int = 7,
        shuffle: bool = True,
    ):
        """
        Function to execute k-fold cross validation on the data with the picked model.
        Model: Either "FCNN" or "OneDCNN"
        """
        models: List[object] = []
        histories: List[object] = []

        kfold = KFold(n_splits=k, shuffle=shuffle)
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
