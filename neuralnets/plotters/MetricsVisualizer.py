import json
import matplotlib.pyplot as plt
import os
import numpy as np


class MetricsVisualizer:
    def __init__(self, results_folder_path):
        self.results_folder_path = results_folder_path
        self.metrics_data = self.load_json()

    def load_json(self):
        metrics_data = {}
        for model_folder in os.listdir(self.results_folder_path):
            model_path = os.path.join(self.results_folder_path, model_folder)
            if os.path.isdir(model_path):
                json_file_path = os.path.join(
                    model_path, model_folder + "_metrics.json"
                )
                if os.path.isfile(json_file_path):
                    with open(json_file_path, "r") as file:
                        data = json.load(file)
                    metrics_data[model_folder] = data
        return metrics_data

    def plot_accuracy(self):
        plt.style.use("ggplot")
        model_names = []
        average_accuracies = []
        stdev_accuracies = []
        for model_name, data in self.metrics_data.items():
            model_names.append(model_name)
            average_accuracies.append(data["average_validation_accuracy"])
            stdev_accuracies.append(data["stdev_validation_accuracy"])
        x_pos = np.arange(len(model_names))
        plt.figure(figsize=(20, 6))
        plt.bar(
            x_pos,
            average_accuracies,
            yerr=stdev_accuracies,
            align="center",
            alpha=0.5,
            ecolor="black",
            capsize=10,
        )
        plt.xticks(x_pos, model_names, fontsize=10)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title("Validation Accuracy", fontsize=16)
        plt.grid(True)
        plt.savefig(os.path.join(self.results_folder_path, "accuracy_plot.png"))
        plt.clf()

    def plot_loss(self):
        plt.style.use("ggplot")
        model_names = []
        average_losses = []
        stdev_losses = []
        for model_name, data in self.metrics_data.items():
            model_names.append(model_name)
            average_losses.append(data["average_validation_loss"])
            stdev_losses.append(data["stdev_validation_loss"])
        x_pos = np.arange(len(model_names))
        plt.figure(figsize=(20, 6))
        plt.bar(
            x_pos,
            average_losses,
            yerr=stdev_losses,
            align="center",
            alpha=0.5,
            ecolor="black",
            capsize=10,
            color="blue",  # Set the color of the bars to blue
        )
        plt.xticks(x_pos, model_names, fontsize=10)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Validation Loss", fontsize=16)
        plt.grid(True)
        plt.savefig(os.path.join(self.results_folder_path, "loss_plot.png"))
        plt.clf()
