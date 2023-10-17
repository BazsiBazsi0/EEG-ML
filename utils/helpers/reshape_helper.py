import numpy as np
from sklearn.preprocessing import minmax_scale
from utils.helpers.one_hot_helper import OneHotHelper
from utils.helpers.smote_helper import SmoteHelper


class ReShapeHelper:
    @staticmethod
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
