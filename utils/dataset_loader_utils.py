import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical
import nn


class FileLoader:
    # TODO: Testing
    def __init__(self, patients=10, subjects=range(60, 70)):
        self.patients = patients
        self.subjects = subjects

    def to_one_hot(self, y):
        # shallow copy to a new array
        y_shallow_copy = y.copy()
        # New unique labels in case of double vals(maybe there are duplicates)
        total_labels = np.unique(y_shallow_copy)

        # Dictionary named encoding for labels
        encoding = {}
        for x in range(len(total_labels)):
            encoding[total_labels[x]] = x
        for x in range(len(y_shallow_copy)):
            y_shallow_copy[x] = encoding[y_shallow_copy[x]]

        return to_categorical(y_shallow_copy)

    def smote_processor(self, x, y):
        sm = SMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_resample(x, y)
        return x_resampled, y_resampled

    def preprocessor(self, x, y):
        # Reshaping to 2D
        x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).astype(
            np.float16
        )

        # Minmax scaling
        x_scaled = minmax_scale(x_reshaped, axis=1).astype(np.float16)

        # One_hot encoding
        y_one_hot = self.to_one_hot(y).astype(np.uintc)

        # Synthetic Minority Oversampling Technique
        x_smote, y_smote = self.smote_processor(x_scaled, y_one_hot)

        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x_smote = np.reshape(
            x_smote, (x_smote.shape[0], x.shape[1], x_smote.shape[1] // x.shape[1])
        )
        x_no_smote = np.reshape(
            x_scaled, (x_scaled.shape[0], x.shape[1], x_scaled.shape[1] // x.shape[1])
        )

        return x_smote, y_smote, x_no_smote, y_one_hot

    def load_saved_files_val(self):
        xs, ys = [], []
        for s in self.subjects:
            # level 0 7, level 1
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
