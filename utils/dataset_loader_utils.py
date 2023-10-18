import numpy as np
from sklearn.preprocessing import minmax_scale
from utils.helpers.one_hot_helper import OneHotHelper
from utils.helpers.smote_helper import SmoteHelper

class FileLoader:
    def __init__(self, patients=10, subjects=range(60, 70)):
        self.patients = patients
        self.subjects = subjects

    def load_saved_files_val(self):
        xs, ys = [], []
        for s in self.subjects:
            # levels loading
            xs.append(
                np.load("dataset/filtered_data/ch_level_1/x_sub_{}.npy".format(s))
            )
            ys.append(
                np.load("dataset/filtered_data/ch_level_1/y_sub_{}.npy".format(s))
            )

        x_concat = np.concatenate(xs)
        y_concat = np.concatenate(ys)

        # Pre-processing the variables
        x_pre, y_pre, x_no_smote_pre, y_no_smote_pre = self.preprocessor(
            x_concat, y_concat
        )

        num_patients = x_pre.shape[0] // self.patients
        x = x_pre.reshape(
            self.patients, num_patients, x_pre.shape[1], x_pre.shape[2]
        ).astype(np.float32)
        y = y_pre.reshape(self.patients, num_patients, y_pre.shape[1]).astype(
            np.float32
        )

        x_no_smote = x_no_smote_pre.reshape(
            self.patients,
            x_no_smote_pre.shape[0] // self.patients,
            x_no_smote_pre.shape[1],
            x_no_smote_pre.shape[2],
        ).astype(np.float32)

        y_no_smote = y_no_smote_pre.reshape(
            self.patients,
            y_no_smote_pre.shape[0] // self.patients,
            y_no_smote_pre.shape[1],
        ).astype(np.float32)

        return x, y, x_no_smote, y_no_smote

    def preprocessor(x: np.ndarray, y: np.ndarray):
        """
        Preprocesses the input data and returns the processed data.

        Parameters:
            x (np.ndarray): The input data.
            y (np.ndarray): The labels of the input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The preprocessed data
        """
        # Reshaping to 2D
        x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).astype(
            np.float16
        )

        # Minmax scaling
        x_scaled = minmax_scale(x_reshaped, axis=1).astype(np.float16)

        # One_hot encoding
        y_one_hot = OneHotHelper.to_one_hot(y).astype(np.uintc)

        # Synthetic Minority Oversampling Technique
        x_smote, y_smote = SmoteHelper.smote_processor(x_scaled, y_one_hot)

        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x_smote = np.reshape(
            x_smote, (x_smote.shape[0], x.shape[1], x_smote.shape[1] // x.shape[1])
        )

        x_no_smote = np.reshape(
            x_scaled, (x_scaled.shape[0], x.shape[1], x_scaled.shape[1] // x.shape[1])
        )

        return x_smote, y_smote, x_no_smote, y_one_hot

