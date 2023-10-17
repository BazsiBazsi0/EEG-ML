import numpy as np
from tensorflow.keras.utils import to_categorical

class OneHotHelper:
    @staticmethod
    def to_one_hot(y: np.ndarray) -> np.ndarray:
        """
        Converts the input array to one-hot encoded array.

        Parameters:
            y (np.ndarray): The input array.

        Returns:
            np.ndarray: The one-hot encoded array.
        """
        # Shallow copy to a new array
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
