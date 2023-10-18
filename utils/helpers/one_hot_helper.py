import numpy as np
from typing import Dict

class OneHotHelper:
    @staticmethod
    def to_one_hot(y: np.ndarray) -> np.ndarray:
        """
        This static method transforms an input numpy array into a one-hot encoded array.
        It identifies unique labels in the array and maps each unique label to a unique integer.
        Then, it uses these mappings to convert the input array into a one-hot encoded array.

        Args:
            y (np.ndarray): The input array to be one-hot encoded.

        Returns:
            np.ndarray: The one-hot encoded version of the input array.
        """
        # Identify unique labels in the array
        total_labels = np.unique(y)

        # Create a dictionary mapping each unique label to a unique integer
        encoding: Dict[int, int] = {label: index for index, label in enumerate(total_labels)}

        # Create an array for one-hot encoding
        one_hot_array = np.zeros((y.size, total_labels.size))

        # Use the dictionary to convert the input array into a one-hot encoded array
        for i in range(len(y)):
            if y[i] not in encoding:
                raise ValueError(f"Unexpected value {y[i]} at index {i}.")
            one_hot_array[i, encoding[y[i]]] = 1

        return one_hot_array
