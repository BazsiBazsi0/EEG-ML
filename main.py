import os
import numpy as np
import fileloader
import modeltrainer
import tensorflow as tf
import utils.config as config
from utils.dataset_download_utils import Downloader
from utils.dataset_utils import DatasetUtils
from utils import logging_utils
from neuralnets.eval.ModelEvaluator import ModelEvaluator
from neuralnets.plotters.DataVisualizer import DataVisualizer
from fileprocessor import FileProcessor

if __name__ == "__main__":

    # Only use the first available gpu/device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Logging setup
    logger = logging_utils.Logger(__name__)

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

    # Data loading, load_level parameter is the level of electrodes to be used(0: lowest, 3: highest)
    load_level = 0

    # Loading the saved files
    x, y = fileloader.FileLoader.load_saved_files(load_level)

    # Pre-processing the variables
    FileProcessor = FileProcessor(x, y)
    x, y, x_smote, y_smote, x_val, y_val = FileProcessor.preprocessor()

    # Data loading and preprocessing done, time to train the model
    # TODO: Document kfold, Plateou LR reduction(0.0001 and try later 0.00001) and later OneCycleScheduler
    histories, models = modeltrainer.NeuralNets.k_fold_validation(
        x,
        y,
        k=4,
        epochs=50,
        model_name="OneDCNN",
        load_level=load_level,
        electrodes=len(config.ch_level[load_level]),
    )

    """
    models = tf.keras.models.load_model(
        "Results/FCNN_ch_level_0/FCNN_ch_level_0_model_0.keras"
    )
    """
    ModelEvaluator.evaluate(
        models,
        histories,
        x_val,
        y_val,
        ["save", "curves", "metrics", "roc_curve"],
        results_dir="Results",
    )
