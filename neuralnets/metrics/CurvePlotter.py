import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class CurvePlotter:
    @staticmethod
    def save_curves(histories, models, results_dir: str = "Results"):
        """
        Save the training and validation curves for multiple models.

        Parameters:
        - histories (list): List of training histories for each model.
        - filename (str): Name of the file to save the curves.

        Returns:
        None
        """
        # Get the model name
        model_name = models[0].name

        # Create a directory for the model under "Results" if it doesn't exist
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Define the file name
        file_name = os.path.join(model_dir, f"{model_name}_level_0_metrics_curves.png")

        num_histories = len(histories)
        fig, axes = plt.subplots(num_histories, 2, figsize=(12, 6 * num_histories))

        for i, history in enumerate(histories):
            acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]

            epochs_range = range(len(acc))

            axes[i, 0].plot(epochs_range, acc, label="train accuracy")
            axes[i, 0].plot(epochs_range, val_acc, label="validation accuracy")
            axes[i, 0].set_title("Accuracy")
            axes[i, 0].set_xlabel("Epoch")
            axes[i, 0].set_ylabel("Accuracy")
            axes[i, 0].legend(loc="lower right")
            axes[i, 0].text(
                len(acc) - 1,
                acc[-1],
                f"End Train Acc: {acc[-1]:.4f}",
                ha="right",
                va="center",
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )
            axes[i, 0].text(
                len(val_acc) - 1,
                val_acc[-1],
                f"End Val Acc: {val_acc[-1]:.4f}",
                ha="right",
                va="center",
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )
            axes[i, 0].grid(True)  # Add grid

            axes[i, 1].plot(epochs_range, loss, label="train loss")
            axes[i, 1].plot(epochs_range, val_loss, label="validation loss")
            axes[i, 1].set_title("Loss")
            axes[i, 1].set_xlabel("Epoch")
            axes[i, 1].set_ylabel("Loss")
            axes[i, 1].legend(loc="upper right")
            axes[i, 1].text(
                len(loss) - 1,
                loss[-1],
                f"End Train Loss: {loss[-1]:.4f}",
                ha="right",
                va="center",
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )
            axes[i, 1].text(
                len(val_loss) - 1,
                val_loss[-1],
                f"End Val Loss: {val_loss[-1]:.4f}",
                ha="right",
                va="center",
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"
                ),
            )

            axes[i, 1].grid(True)  # Add grid

        fig.tight_layout()
        plt.savefig(file_name)

    @staticmethod
    def plot_roc_curve(models, x, y, results_dir: str = "Results"):
        """
        Plot the ROC curves for multiple models.

        Parameters:
        - models (list): List of models to plot the ROC curves for.
        - x (numpy.ndarray): Test data.
        - y (numpy.ndarray): Test labels.

        Returns:
        None
        """
        num_models = len(models)
        num_rows = (num_models + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))

        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        model_name = models[0].name
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        file_name = os.path.join(model_dir, model_name)

        for i, model in enumerate(models):
            row = i // 2
            col = i % 2

            labels = ["Left", "Right", "Fists", "Rest", "Feet"]
            y_pred = model.predict(x)
            auc = []
            auc_details = f"{file_name}_AUC_details.txt"
            with open(auc_details, "a") as file:
                file.write(f"For model {i}:\n")

            for j in range(5):
                fpr, tpr, _ = roc_curve(y[:, j], y_pred[:, j])
                axes[row, col].plot(fpr, tpr, label=labels[j])
                axes[row, col].legend(loc="lower right")
                auc.append(round(roc_auc_score(y[:, j], y_pred[:, j]), 4))
                with open(auc_details, "a") as file:
                    file.write(f"AUC for class {j} is {auc[j]}\n")

            axes[row, col].set_xlabel("False Positive Rate")
            axes[row, col].set_ylabel("True Positive Rate")
            axes[row, col].set_title("ROC, AUC=" + str(auc))

        plt.savefig(f"{file_name}_AUC_curves.png")
