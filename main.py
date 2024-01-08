import os
import numpy as np
import fileloader
import nn
from utils.dataset_download_utils import Downloader
from utils.dataset_utils import DatasetUtils

if __name__ == "__main__":

    # Only use the first available gpu/device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initialize the download class
    downloader = Downloader(
        url="https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip"
    )

    # Execute the download
    downloader.download()

    # Download complete, time to load the data
    # Initialize the dataset utils class
    dataset_utils = DatasetUtils()

    # Generate the dataset from the data we have downloaded
    dataset_utils.generate()

    # Data loading
    # TODO: Break apart the loader to a processor and a loader
    x, y, x_no_smote, y_no_smote = fileloader.FileLoader.load_saved_files()
    print("Shape of x: ", np.shape(x))
    print("Instances of classes before SMOTE: ", y_no_smote.sum(axis=1).sum(axis=0))
    print("Instances of classes after SMOTE: ", y.sum(axis=1).sum(axis=0))

    # Execute leave one out validation with a predefined model
    history, model, acc, avgAcc = nn.NeuralNets.loo(x, y)

    # model = nn.NeuralNets.one_d_cnn_multi()
    # nn.NeuralNets.generator_processor(model, x_no_smote, y_no_smote)
    # models, history_loo, acc, avgAcc = nn.NeuralNets.leave_one_out(x_no_smote, y_no_smote)

    # Accuracy saving
    with open("avg accuracy.txt", "a") as f:
        f.write(str(avgAcc) + "\n")

    print("Final avg accuracy:", avgAcc)

    # x, y, x_no_smote, y_no_smote = fileloader.FileLoader.load_saved_files()
    # nn.NeuralNets.metrics_generation(model, x_no_smote, y_no_smote, "1DCNN")
    # nn.NeuralNets.plot_roc_curve(model, x_no_smote, y_no_smote, "1DCNN")

    # Alternate metrics generation
    """nn.NeuralNets.metrics_generation(models['FCN'], x_no_smote, y_no_smote)
    nn.NeuralNets.plot_roc_curve(models['FCN'], x_no_smote, y_no_smote, "FCN.png")"""
