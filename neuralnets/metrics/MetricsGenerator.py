import os
import tensorflow as tf
import json
import numpy as np
from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix
import pickle


class MetricsGenerator:
    """
    A class for generating metrics for neural network models.
    """

    @staticmethod
    def evaluate(
        models: List[tf.keras.Model],
        histories: List[Dict],
        X_val: np.ndarray,
        y_val: np.ndarray,
        tasks: List[str],
        results_dir: str = "Results",
    ):
        """
        Evaluates the models based on the specified tasks.

        Parameters
        ----------
        models : List[tf.keras.Model]
            The models to be evaluated.
        histories : List[Dict]
            The histories of the models.
        X_val : np.ndarray
            The validation data.
        y_val : np.ndarray
            The validation labels.
        tasks : List[str]
            The tasks to be performed. Valid tasks are "save", "curves", "metrics", and "roc_curve".
        results_dir : str, optional
            The directory where the results will be saved (default is "Results").
        """
        if "save" in tasks:
            MetricsGenerator.save_models(histories, models, results_dir)
        if "curves" in tasks:
            MetricsGenerator.save_curves(histories, models, results_dir)
        if "metrics" in tasks:
            MetricsGenerator.save_metrics(histories, models, X_val, y_val, results_dir)
        if "roc_curve" in tasks:
            MetricsGenerator.plot_roc_curve(models, X_val, y_val, results_dir)

    @staticmethod
    def save_metrics(
        histories: List[tf.keras.callbacks.History],
        models: List[tf.keras.Model],
        x: tf.Tensor,
        y: tf.Tensor,
        results_dir: str,
    ) -> None:
        metrics = []
        classification_reports = []
        confusion_matrices = []

        for i, history in enumerate(histories):
            model = models[i]
            model_name = model.name

            # Create a directory for the model under "Results" if it doesn't exist
            model_dir = os.path.join(results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Define the file name
            file_name = os.path.join(model_dir, f"{model_name}_metrics.json")

            y_predict = model.predict(x)
            y_test_class = tf.argmax(y, axis=1)
            y_predicted_classes = tf.argmax(y_predict, 1)

            classification_report_data = classification_report(
                y_test_class,
                y_predicted_classes,
                target_names=["left", "right", "fists", "rest", "feet"],
                zero_division=1,
                output_dict=True,
            )

            accuracy = history.history["val_accuracy"][-1]
            loss = history.history["val_loss"][-1]

            metrics.append({"accuracy": float(accuracy), "loss": float(loss)})
            classification_reports.append(classification_report_data)

            confusion_matrix_data = confusion_matrix(
                y_test_class, y_predicted_classes
            ).tolist()
            confusion_matrices.append(confusion_matrix_data)

        average_accuracy = np.mean([metric["accuracy"] for metric in metrics])
        average_loss = np.mean([metric["loss"] for metric in metrics])

        metrics_data = {
            "average_validation_accuracy": float(average_accuracy),
            "average_validation_loss": float(average_loss),
            "stdev_validation_accuracy": float(
                np.std([metric["accuracy"] for metric in metrics])
            ),
            "stdev_validation_loss": float(
                np.std([metric["loss"] for metric in metrics])
            ),
            "individual_metrics": metrics,
            "classification_reports": classification_reports,
            "confusion_matrices": confusion_matrices,
        }

        with open(file_name, "w") as file:
            json.dump(metrics_data, file, indent=4)

    @staticmethod
    def save_models(
        histories: List[Dict], models: List[tf.keras.Model], results_dir: str
    ) -> None:
        for i, (model, history) in enumerate(zip(models, histories)):
            model_name = model.name
            model_dir = os.path.join(results_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save the model
            file_name = os.path.join(model_dir, f"{model_name}_model_{i}.keras")
            model.save(file_name)

            # Save the history
            history_file_name = os.path.join(
                model_dir, f"{model_name}_history_{i}.pickle"
            )
            with open(history_file_name, "wb") as f:
                pickle.dump(history.history, f)
        # Save the model summary
        MetricsGenerator.save_model_summary(
            models[0], os.path.join(model_dir, f"{model_name}_model_summary.txt")
        )

    @staticmethod
    def save_model_summary(model, filepath):
        with open(filepath, "w") as f:
            # Redirect the output of model.summary() to the file
            model.summary(line_length=120, print_fn=lambda x: f.write(x + "\n"))
