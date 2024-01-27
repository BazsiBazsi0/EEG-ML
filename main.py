import os
import numpy as np
import fileloader
import nn
from utils.dataset_download_utils import Downloader
from utils.dataset_utils import DatasetUtils

if __name__ == "__main__":

    # Only use the first available gpu/device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    # TODO: Proper logging
    print("Shape of x: ", np.shape(x))
    print("Instances of classes before SMOTE: ", y_no_smote.sum(axis=1).sum(axis=0))
    print("Instances of classes after SMOTE: ", y.sum(axis=1).sum(axis=0))

    # Data loading and preprocessing done, time to train the model
    # Execute leave one out(k_fold n=10) validation with a predefined model
    # TODO: Document kfold, Plateou LR reduction(0.0001 and try later 0.00001) and later OneCycleScheduler
    history, model, acc, avgAcc = nn.NeuralNets.k_fold_validation(x, y)

    # TODO: Proper logging
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
