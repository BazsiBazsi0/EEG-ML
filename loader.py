import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical


class FileLoader:
    # if I want to switch this to another method here are the possibilities:
    # use sklearn onehot, this could cause problems down the line, need to investigate
    # use tf.one_hot but this doesn't preserve the labels, since it's a integer/tensor operation
    # use this with minor improvements as this already uses the .to_categorical method
    @staticmethod
    def to_one_hot(y):
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

    # Synthetic Minority Oversampling Technique
    # This balances the imbalance between 'rest' (re) and the other classes
    # more reading: https://imbalanced-learn.org/stable/over_sampling.html
    @staticmethod
    def smote_processor(x, y):
        sm = SMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_resample(x, y)
        return x_resampled, y_resampled

    # preprocessor, receives x and y in a concatenated form, transforms data into a 2D format
    @staticmethod
    def preprocessor(x, y):
        # Reshaping to 2D
        x_reshaped = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        # Minmax scaling
        x_scaled = minmax_scale(x_reshaped, axis=1)

        # One_hot encoding
        y_one_hot = FileLoader.to_one_hot(y)

        # Synthetic Minority Oversampling Technique
        x_smote, y_smote = FileLoader.smote_processor(x_scaled, y_one_hot)

        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x = x_smote.reshape(x_smote.shape[0], x.shape[1], x_smote.shape[1] // x.shape[1])
        x_no_smote = x_scaled.reshape(x_scaled.shape[0], x.shape[1], x_scaled.shape[1] // x.shape[1])
        y_no_smote = y_one_hot

        return x, y_smote, x_no_smote, y_no_smote

    @staticmethod
    def load_saved_files(patients):
        # load the saved files
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in range(1, patients+1) if n not in exclude]
        xs = []
        ys = []
        data_x = []
        data_y = []
        for s in subjects:
            xs.append(np.load("legacy_generator/all_electrodes_50_patients/" + "x" + "_sub_" + str(s) + ".npy"))
            ys.append(np.load("legacy_generator/all_electrodes_50_patients/" + "y" + "_sub_" + str(s) + ".npy"))

        data_x.append(np.concatenate(xs))
        data_y.append(np.concatenate(ys))

        x_concat = np.concatenate(data_x)
        y_concat = np.concatenate(data_y)

        # Pre-processing the variables
        x_pre, y_pre, x_no_smote_pre, y_no_smote_pre = FileLoader.preprocessor(x_concat, y_concat)

        # Reshaping X into 4D array, Y into 3D array, per patient basis
        # Structure:
        #   x = Patients x Epochs x Channels X Datapoints
        #   y = Patients x Epochs x One-hot Labels
        x = x_pre.reshape(patients, x_pre.shape[0]//patients, x_pre.shape[1], x_pre.shape[2])
        y = y_pre.reshape(patients, y_pre.shape[0]//patients, y_pre.shape[1])

        x_no_smote = x_no_smote_pre.reshape(patients, x_no_smote_pre.shape[0]//patients, x_no_smote_pre.shape[1], x_no_smote_pre.shape[2])
        y_no_smote = y_no_smote_pre.reshape(patients, y_no_smote_pre.shape[0]//patients, y_no_smote_pre.shape[1])

        return x, y, x_no_smote, y_no_smote
