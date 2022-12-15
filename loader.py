import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical
import nn


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
        x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).astype(np.float16)

        # Minmax scaling
        x_scaled = minmax_scale(x_reshaped, axis=1).astype(np.float16)

        # One_hot encoding
        y_one_hot = FileLoader.to_one_hot(y).astype(np.uintc)

        # Synthetic Minority Oversampling Technique
        x_smote, y_smote = FileLoader.smote_processor(x_scaled, y_one_hot)

        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x_smote = np.reshape(x_smote, (x_smote.shape[0], x.shape[1], x_smote.shape[1] // x.shape[1]))
        x_no_smote = np.reshape(x_scaled, (x_scaled.shape[0], x.shape[1], x_scaled.shape[1] // x.shape[1]))

        return x_smote, y_smote, x_no_smote, y_one_hot

    @staticmethod
    def data_equalizer(x, y):
        # The purpose of this function is to slim down x or y, so they have equal sample sizes
        # This is destructive data processing technique
        # I saw somewhere during loading that the slice down the rest(baseline) might worth investigating

        # Finding the minium occurrence tests, this is the data we need equalize all the other test sample numbers
        # This is not the number of all tests permitted, but the total number of test within a class for each person
        min_test = y.sum(axis=1).sum(axis=0).min() / len(x)  # only needed when per persons basis the picking

        # The lowest class occurrence is the first one. 17 occurrence in every patient. 17 * 5(there are 5 classes) = 85
        # Since we have to traverse the original array and levea out all classes that have higher occurance than 17,
        # we are going to create an array that is the same size of the original and later delete the zero values
        # Later I'm are going to create it dynamically based on the shape of the incoming array

        # Update: it seems like the np.empty or np.zeros is good enough so it might not need deletion after creation
        new_x = np.empty((10, 35, 64, 641))
        new_y = np.empty((10, 35, 5))
        classes_count = np.zeros((10, 5))
        for i in range(len(new_y)):
            for t in range(len(new_y[i])):
                for z in range(len(new_y[i][t])):
                    if y[i][t][z] == 1 and classes_count[i][z] < 7:
                        new_x[i][t] = x[i][t]
                        new_y[i][t] = y[i][t]
                        classes_count[i][z] += 1

        return new_x, new_y

    @staticmethod
    def load_saved_files():
        # The output for this function is patients x epochs x channels x datapoints
        # for x_no smote with 1 patient its 1 x 160 x 64 x 641 ( one patient with 160 epoch, 64 chs, and 641 datapoints)
        # for y_no_smote with 1 patient its 1 x 160 x 5 (1 patient with 160 epoch and 5 classes in each epoch)

        # load the saved files
        patients = 10
        subjects = []
        for i in range(patients):
            # Generate a random number between 0 and 103
            random_number = np.random.randint(0, 103)
            # Check if the random number is in the list of exceptions
            if random_number in [38, 88, 89, 92, 100, 104, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
                # If it is, generate a new random number
                random_number = np.random.randint(0, 103)
            # Add the random number to the list
            subjects.append(random_number)

        xs, ys = [], []
        for s in subjects:
            # level 0 7, level 1
            xs.append(np.load("generator/all_electrodes_103_patients_ch_level_1/x_sub_{}.npy".format(s)))
            ys.append(np.load("generator/all_electrodes_103_patients_ch_level_1/y_sub_{}.npy".format(s)))

        x_concat = np.concatenate(xs)
        y_concat = np.concatenate(ys)

        # Pre-processing the variables
        x_pre, y_pre, x_no_smote_pre, y_no_smote_pre = FileLoader.preprocessor(x_concat, y_concat)

        # Reshaping X into 4D array, Y into 3D array, per patient basis
        # Structure:
        #   x = Patients x Epochs x Channels X Datapoints
        #   y = Patients x Epochs x One-hot Labels
        num_patients = x_pre.shape[0] // patients
        x = x_pre.reshape(patients, num_patients, x_pre.shape[1], x_pre.shape[2]).astype(np.float32)
        y = y_pre.reshape(patients, num_patients, y_pre.shape[1]).astype(np.float32)

        x_no_smote = x_no_smote_pre.reshape(patients, x_no_smote_pre.shape[0] // patients, x_no_smote_pre.shape[1],
                                            x_no_smote_pre.shape[2]).astype(np.float32)
        y_no_smote = y_no_smote_pre.reshape(patients,  y_no_smote_pre.shape[0] // patients, y_no_smote_pre.shape[1]).astype(np.float32)

        x_fft, y_fft = nn.NeuralNets.fft_processor(x_no_smote, y_no_smote)
        return x_fft, y_fft

    @staticmethod
    def load_saved_files_val():
        # The output for this function is patients x epochs x channels x datapoints
        # for x_no smote with 1 patient its 1 x 160 x 64 x 641 ( one patient with 160 epoch, 64 chs, and 641 datapoints)
        # for y_no_smote with 1 patient its 1 x 160 x 5 (1 patient with 160 epoch and 5 classes in each epoch)

        # load the saved files
        patients = 10
        subjects = range(60, 70)

        xs, ys = [], []
        for s in subjects:
            # level 0 7, level 1
            xs.append(np.load("generator/all_electrodes_103_patients_ch_level_1/x_sub_{}.npy".format(s)))
            ys.append(np.load("generator/all_electrodes_103_patients_ch_level_1/y_sub_{}.npy".format(s)))

        x_concat = np.concatenate(xs)
        y_concat = np.concatenate(ys)

        # Pre-processing the variables
        x_pre, y_pre, x_no_smote_pre, y_no_smote_pre = FileLoader.preprocessor(x_concat, y_concat)

        # Reshaping X into 4D array, Y into 3D array, per patient basis
        # Structure:
        #   x = Patients x Epochs x Channels X Datapoints
        #   y = Patients x Epochs x One-hot Labels
        num_patients = x_pre.shape[0] // patients
        x = x_pre.reshape(patients, num_patients, x_pre.shape[1], x_pre.shape[2]).astype(np.float32)
        y = y_pre.reshape(patients, num_patients, y_pre.shape[1]).astype(np.float32)

        x_no_smote = x_no_smote_pre.reshape(patients, x_no_smote_pre.shape[0] // patients, x_no_smote_pre.shape[1],
                                            x_no_smote_pre.shape[2]).astype(np.float32)
        y_no_smote = y_no_smote_pre.reshape(patients,  y_no_smote_pre.shape[0] // patients, y_no_smote_pre.shape[1]).astype(np.float32)

        return x, y, x_no_smote, y_no_smote