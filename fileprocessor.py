import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


class FileProcessor:
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
    # TODO: Somehow the return shape is 4015 instead of 4150 for x_resampled. Investigate
    # Theory: sampling_strategy needs to be set (eg to 1000) so that it doesn't create class imbalance
    # The actual values of X doesnt seem to be correct.
    @staticmethod
    def smote_processor(x, y):
        sm = SMOTE(
            random_state=42  # ,
            # sampling_strategy={0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000},
        )
        x_resampled, y_resampled = sm.fit_resample(x, y)
        return x_resampled, y_resampled

    # preprocessor, receives x and y in a concatenated form, transforms data into a 2D format
    @staticmethod
    def preprocessor(x, y):
        # Reshaping to 2D
        # Needs y as 1600, 5 amd x as 1600, *(13461)
        x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).astype(
            np.float16
        )

        # Normalize the data
        # TODO: Docu
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x_reshaped)

        # Minmax scaling
        x_scaled = minmax_scale(x_normalized, axis=1, feature_range=(-1, 1)).astype(
            np.float16
        )

        # One_hot encoding
        y_one_hot = FileProcessor.to_one_hot(y)

        # Synthetic Minority Oversampling Technique
        x_smote, y_smote = FileProcessor.smote_processor(x_scaled, y_one_hot)

        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x_smote = np.reshape(
            x_smote, (x_smote.shape[0], x.shape[1], x_smote.shape[1] // x.shape[1])
        )
        x_no_smote = np.reshape(
            x_scaled, (x_scaled.shape[0], x.shape[1], x_scaled.shape[1] // x.shape[1])
        )

        return x_smote, y_smote, x_no_smote, y_one_hot

    @staticmethod
    def data_equalizer(x, y):
        # The purpose of this function is to slim down x or y, so they have equal sample sizes
        # This is destructive data processing technique
        # I saw somewhere during loading that the slice down the rest(baseline) might worth investigating

        # Finding the minium occurrence tests, this is the data we need equalize all the other test sample numbers
        # This is not the number of all tests permitted, but the total number of test within a class for each person
        min_test = y.sum(axis=1).sum(axis=0).min() / len(
            x
        )  # only needed when per persons basis the picking

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
