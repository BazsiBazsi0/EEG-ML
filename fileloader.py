import numpy as np
from imblearn.over_sampling import SMOTE
import config
from fileprocessor import FileProcessor


class FileLoader:
    @staticmethod
    def load_saved_files():
        # TODO find a way to load all patients at the same time. Data is king.
        # The output for this function is patients x epochs x channels x datapoints
        # for x_no smote with 1 patient its 1 x 160 x 64 x 641 ( one patient with 160 epoch, 64 chs, and 641 datapoints)
        # for y_no_smote with 1 patient its 1 x 160 x 5 (1 patient with 160 epoch and 5 classes in each epoch)

        # load ten patients randomly from the saved files
        # TODO: load random patients not preselected ones
        patients = 10
        subjects = range(1, 11)
        for i in range(patients):
            # Generate a random number between 0 and 103
            random_number = np.random.randint(0, 103)
            # Check if the random number is in the list of exceptions
            if random_number in config.excluded:
                # If it is, generate a new random number
                random_number = np.random.randint(0, 103)
                continue
            # Add the random number to the list
            # subjects.append(random_number)

        xs, ys = [], []
        for s in subjects:
            xs.append(
                np.load("dataset/filtered_data/ch_level_2/x_sub_{}.npy".format(s))
            )
            ys.append(
                np.load("dataset/filtered_data/ch_level_2/y_sub_{}.npy".format(s))
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
