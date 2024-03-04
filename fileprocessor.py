import numpy as np
from numpy import ndarray
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


class FileProcessor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_resampled = None
        self.y_resampled = None

    def preprocessor(self):
        # Reshaping to 2D to complete basic preprocessing
        x_reshaped = np.reshape(
            self.x, (self.x.shape[0], self.x.shape[1] * self.x.shape[2])
        )

        # Normalize the data
        # TODO: Document changes
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(x_reshaped)

        # Minmax scaling
        x_scaled = minmax_scale(x_normalized, axis=1, feature_range=(-1, 1))

        # One_hot encoding
        y_one_hot = FileProcessor.to_one_hot(self.y)

        # Creating and subtracting the validation dataset before applying smote

        split_index = int(len(x_scaled) * 0.8)

        # This way the validation dataset doesn't poisons our training set any way
        x_val = x_scaled[split_index:]
        y_val = y_one_hot[split_index:]
        x_scaled = x_scaled[:split_index]
        y_one_hot = y_one_hot[:split_index]

        # x_scaled, y_one_hot = FileProcessor.remove_majority_class(x_scaled, y_one_hot)

        # Synthetic Minority Oversampling Technique
        x_smote, y_smote = FileProcessor.smote_processor(x_scaled, y_one_hot)

        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x_smote = np.reshape(
            x_smote,
            (x_smote.shape[0], x_smote.shape[1] // self.x.shape[1], self.x.shape[1]),
        )
        x_no_smote = np.reshape(
            x_scaled,
            (x_scaled.shape[0], x_scaled.shape[1] // self.x.shape[1], self.x.shape[1]),
        )
        x_val = np.reshape(
            x_val, (x_val.shape[0], x_val.shape[1] // self.x.shape[1], self.x.shape[1])
        )

        # Equalize the data
        x_no_smote, y_one_hot = FileProcessor.equalize_samples(x_no_smote, y_one_hot)

        return x_no_smote, y_one_hot, x_smote, y_smote, x_val, y_val

    @staticmethod
    def to_one_hot(y):
        # shallow copy to a new array
        y_copy = y.copy()
        # New unique labels in case of double vals(maybe there are duplicates)
        total_labels = np.unique(y_copy)

        # Dictionary named encoding for labels
        encoding = {}
        for x in range(len(total_labels)):
            encoding[total_labels[x]] = x
        for x in range(len(y_copy)):
            y_copy[x] = encoding[y_copy[x]]

        return to_categorical(y_copy)

    @staticmethod
    def smote_processor(x, y):
        """SMOTE Data augmentation
        print("Class instances before SMOTE: ", y.sum(axis=0))
        print("Class instances after SMOTE: ", y_smote.sum(axis=0))
        """
        sm = SMOTE(random_state=42)
        x_resampled, y_resampled = sm.fit_resample(x, y)
        return x_resampled, y_resampled

    @staticmethod
    def equalize_samples(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        """
        Equalizes the data by reducing the sample sizes of the class that has too many samples.

        Args:
            x: The input data array.
            y: The labels array.

        Returns:
            The equalized data as a tuple of arrays (new_x, new_y).
        """
        # Calculate the number of samples in each class
        class_counts = y.sum(axis=0)

        # Find the minimum number of samples in any class
        min_samples = int(np.min(class_counts))

        # Create masks for each class
        class_masks = [y[:, i] == 1 for i in range(y.shape[1])]

        # Apply the masks to x and y, and limit the number of samples in each class
        new_x = np.concatenate(
            [x[class_mask][:min_samples] for class_mask in class_masks]
        )
        new_y = np.concatenate(
            [y[class_mask][:min_samples] for class_mask in class_masks]
        )

        return new_x, new_y

    @staticmethod
    def remove_majority_class(x: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        """
        Removes the majority class completely from the data.

        Args:
            x: The input data array.
            y: The labels array.

        Returns:
            The modified data as a tuple of arrays (new_x, new_y).
        """
        # Calculate the number of samples in each class
        class_counts = y.sum(axis=0)

        # Find the index of the majority class
        majority_class_index = np.argmax(class_counts)

        # Create masks for each class
        class_masks = [y[:, i] == 1 for i in range(y.shape[1])]

        # Remove the majority class
        class_masks.pop(majority_class_index)

        # Apply the masks to x and y
        new_x = np.concatenate([x[class_mask] for class_mask in class_masks])
        new_y = np.concatenate([y[class_mask] for class_mask in class_masks])

        # Remove the majority class from the second dimension of new_y
        new_y = np.delete(new_y, majority_class_index, axis=1)

        return new_x, new_y
