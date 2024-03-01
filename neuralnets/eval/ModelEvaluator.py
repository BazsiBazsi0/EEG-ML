from typing import List, Dict
import tensorflow as tf
import numpy as np
from neuralnets.metrics.MetricsGenerator import MetricsGenerator
from neuralnets.metrics.CurvePlotter import CurvePlotter


class ModelEvaluator:
    """
    A class used to evaluate models.

    ...

    Methods
    -------
    evaluate(models: List[tf.keras.Model], histories: List[Dict],
        X_val: np.ndarray, y_val: np.ndarray, tasks: List[str], results_dir: str = "Results")
        Evaluates the models based on the specified tasks.
        Example usage:
        ModelEvaluator.evaluate(models, histories, X_val, y_val, ["save", "curves", "metrics", "roc_curve"], results_dir="Results")

    save_models(models: List[tf.keras.Model], results_dir: str = "Results")
        Saves the models to the specified directory.

    save_curves(histories: List[Dict], models: List[tf.keras.Model], results_dir: str = "Results")
        Saves the learning curves of the models to the specified directory.

    save_metrics(histories: List[Dict], models: List[tf.keras.Model], X_val: np.ndarray, y_val: np.ndarray, results_dir: str = "Results")
        Saves the metrics of the models to the specified directory.

    plot_roc_curve(models: List[tf.keras.Model], X_val: np.ndarray, y_val: np.ndarray, results_dir: str = "Results")
        Plots the ROC curve of the models and saves it to the specified directory.
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
            ModelEvaluator.save_models(histories, models, results_dir)
        if "curves" in tasks:
            ModelEvaluator.save_curves(histories, models, results_dir)
        if "metrics" in tasks:
            ModelEvaluator.save_metrics(histories, models, X_val, y_val, results_dir)
        if "roc_curve" in tasks:
            ModelEvaluator.plot_roc_curve(models, X_val, y_val, results_dir)

    @staticmethod
    def save_models(
        histories, models: List[tf.keras.Model], results_dir: str = "Results"
    ):
        """
        Saves the models to the specified directory.

        Parameters
        ----------
        histories : List[Dict]
            The histories of the models.
        models : List[tf.keras.Model]
            The models to be saved.
        results_dir : str, optional
            The directory where the models will be saved (default is "Results").
        """
        MetricsGenerator.save_models(histories, models, results_dir)

    @staticmethod
    def save_curves(
        histories: List[Dict],
        models: List[tf.keras.Model],
        results_dir: str = "Results",
    ):
        """
        Saves the learning curves of the models to the specified directory.

        Parameters
        ----------
        histories : List[Dict]
            The histories of the models.
        models : List[tf.keras.Model]
            The models whose learning curves will be saved.
        results_dir : str, optional
            The directory where the learning curves will be saved (default is "Results").
        """
        CurvePlotter.save_curves(histories, models, results_dir)

    @staticmethod
    def save_metrics(
        histories: List[Dict],
        models: List[tf.keras.Model],
        X_val: np.ndarray,
        y_val: np.ndarray,
        results_dir: str = "Results",
    ):
        """
        Saves the metrics of the models to the specified directory.

        Parameters
        ----------
        histories : List[Dict]
            The histories of the models.
        models : List[tf.keras.Model]
            The models whose metrics will be saved.
        X_val : np.ndarray
            The validation data.
        y_val : np.ndarray
            The validation labels.
        results_dir : str, optional
            The directory where the metrics will be saved (default is "Results").
        """
        MetricsGenerator.save_metrics(histories, models, X_val, y_val, results_dir)

    @staticmethod
    def plot_roc_curve(
        models: List[tf.keras.Model],
        X_val: np.ndarray,
        y_val: np.ndarray,
        results_dir: str = "Results",
    ):
        """
        Plots the ROC curve of the models and saves it to the specified directory.

        Parameters
        ----------
        models : List[tf.keras.Model]
            The models whose ROC curve will be plotted.
        X_val : np.ndarray
            The validation data.
        y_val : np.ndarray
            The validation labels.
        results_dir : str, optional
            The directory where the ROC curve will be saved (default is "Results").
        """
        CurvePlotter.plot_roc_curve(models, X_val, y_val, results_dir)
