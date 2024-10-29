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
        """
        Preprocess the input data for machine learning tasks.

        This method performs several preprocessing steps on the input data:
        1. Reshapes the input data from 3D to 2D
        2. Normalizes the data using StandardScaler
        3. Applies min-max scaling
        4. Performs one-hot encoding on the labels
        5. Splits the data into training and validation sets
        6. Reshapes the data back to 3D format
        7. Equalizes the samples across classes

        Returns:
            tuple: A tuple containing four elements:
                - self.x (numpy.ndarray): Preprocessed training input data
                - self.y (numpy.ndarray): Preprocessed training labels
                - x_val (numpy.ndarray): Preprocessed validation input data
                - y_val (numpy.ndarray): Preprocessed validation labels

        Note:
            This method modifies the `x` and `y` attributes of the FileProcessor instance.
        """
        # Reshaping to 2D to complete basic preprocessing
        x_shape = self.x.shape
        self.x = np.reshape(
            self.x, (self.x.shape[0], self.x.shape[1] * self.x.shape[2])
        )

        # Normalize the data
        # TODO: Document changes
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

        # Minmax scaling
        self.x = minmax_scale(self.x, axis=1, feature_range=(-1, 1), copy=False)

        # One_hot encoding
        self.y = FileProcessor.to_one_hot(self.y)

        # Creating and subtracting the validation dataset before applying smote
        split_index = int(len(self.x) * 0.8)

        # This way the validation dataset doesn't poisons our training set any way
        x_val = self.x[split_index:]
        y_val = self.y[split_index:]
        self.x = self.x[:split_index]
        self.y = self.y[:split_index]

        # x_scaled, y_one_hot = FileProcessor.remove_majority_class(x_scaled, y_one_hot)

        # Synthetic Minority Oversampling Technique
        # self.x, self.y = FileProcessor.smote_processor(self.x, self.y)

        """
        # two 2d empty arrays
        x_smote, y_smote = np.ones((0, self.x.shape[1])), np.ones((0, self.y.shape[1]))
        # Reshaping back into the orig 3D format
        # Structure: Epochs x Channels x Datapoints
        x_smote = np.reshape(
            x_smote,
            (x_smote.shape[0], x_smote.shape[1] // x_shape[1], x_shape[1]),
        )
        """
        # Commented out smote processing to save memory
        # x_smote, y_smote = 0, 0
        self.x = np.reshape(
            self.x,
            (self.x.shape[0], self.x.shape[1] // x_shape[1], x_shape[1]),
        )
        x_val = np.reshape(
            x_val, (x_val.shape[0], x_val.shape[1] // x_shape[1], x_shape[1])
        )

        # Equalize the data
        self.x, self.y = FileProcessor.equalize_samples(self.x, self.y)

        return self.x, self.y, x_val, y_val

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
        x, y = sm.fit_resample(x, y)
        return x, y

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
