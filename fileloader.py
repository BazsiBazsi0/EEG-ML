import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import config
from fileprocessor import FileProcessor
from typing import List
import os


class FileLoader:
    @staticmethod
    def load_saved_files():
        # TODO find a way to load all patients at the same time. Data is king.
        # The output for this function is patients x epochs x channels x datapoints
        # for x_no smote with 1 patient its 1 x 160 x 64 x 641 ( one patient with 160 epoch, 64 chs, and 641 datapoints)
        # for y_no_smote with 1 patient its 1 x 160 x 5 (1 patient with 160 epoch and 5 classes in each epoch)

        # Set the paths for the dataset and filtered data
        dataset_path: str = config.dataset_path
        filtered_data_path: str = os.path.join(
            dataset_path, "filtered_data", "ch_level_0"
        )
        # Get a list of all x files in the filtered data path
        x_files: List[str] = [
            file for file in os.listdir(filtered_data_path) if file.startswith("x_")
        ]
        # Determine the number of patients based on the number of x files
        patients_full: int = len(x_files)
        # Create a list of subjects from 1 to the number of patients
        subjects: List[int] = list(range(1, patients_full + 1))
        # Get a list of excluded subjects from the config file
        excluded_subjects: List[int] = config.excluded
        # Remove the excluded subjects from the list of subjects
        subjects = set(set(subjects) - set(excluded_subjects))
        patients: int = len(subjects)

        xs, ys = [], []
        for s in subjects:
            xs.append(
                np.load("dataset/filtered_data/ch_level_0/x_sub_{}.npy".format(s))
            )
            ys.append(
                np.load("dataset/filtered_data/ch_level_0/y_sub_{}.npy".format(s))
            )

        x_concat = np.concatenate(xs)
        y_concat = np.concatenate(ys)

        # Pre-processing the variables
        x_pre, y_pre, x_no_smote_pre, y_no_smote_pre = FileProcessor.preprocessor(
            x_concat, y_concat
        )

        # Calculate the number of extra data points
        extra_data_points_x = x_pre.shape[0] % patients
        extra_data_points_y = y_pre.shape[0] % patients

        # Trim the arrays if necessary because smote overdid resampling
        if extra_data_points_x > 0:
            x_pre = x_pre[:-extra_data_points_x]
        if extra_data_points_y > 0:
            y_pre = y_pre[:-extra_data_points_y]

        # Reshaping X into 4D array, Y into 3D array, per patient basis
        # Structure:
        #   x = Patients x Epochs x Channels X Datapoints
        #   y = Patients x Epochs x One-hot Labels
        # TODO: sometimes the class distribution produces uneven results
        # and the reshape function fails. Find a way to fix this drop the reshape
        x = x_pre.reshape(
            patients, x_pre.shape[0] // patients, x_pre.shape[1], x_pre.shape[2]
        ).astype(np.float32)
        y = y_pre.reshape(patients, x_pre.shape[0] // patients, y_pre.shape[1]).astype(
            np.float32
        )

        x_no_smote = x_no_smote_pre.reshape(
            patients,
            x_no_smote_pre.shape[0] // patients,
            x_no_smote_pre.shape[1],
            x_no_smote_pre.shape[2],
        ).astype(np.float32)
        y_no_smote = y_no_smote_pre.reshape(
            patients, y_no_smote_pre.shape[0] // patients, y_no_smote_pre.shape[1]
        ).astype(np.float32)

        return x, y, x_no_smote, y_no_smote
