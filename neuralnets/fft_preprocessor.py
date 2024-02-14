from imblearn.over_sampling import SMOTE
import numpy as np
from typing import Tuple, List, Any


class FFTProcessor:
    """
    A class that applies Fast Fourier Transform (FFT) and SMOTE to process EEG data.

    Attributes:
        smote (SMOTE): The SMOTE object used for oversampling.

    Methods:
        __init__(self, random_state: int = 42) -> None:
            Initializes the FFTProcessor object.

        fit_resample(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Applies FFT and SMOTE to the training data.

    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initializes the FFTProcessor object.

        Args:
            random_state (int): The random seed for reproducibility.

        """
        self.smote = SMOTE(random_state=random_state)

    def fit_resample(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies FFT and SMOTE to the training data.

        Args:
            x_train (np.ndarray): The input training data.
            y_train (np.ndarray): The target training data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The processed input data and target data.

        """
        # Apply Fast Fourier Transform to the training data
        x_train_fft: np.ndarray = np.fft.fft(x_train)

        y_2d: np.ndarray = y_train.reshape((-1, y_train.shape[2]))

        magnitude: np.ndarray = np.abs(x_train_fft)
        phase: np.ndarray = np.angle(x_train_fft)
        magnitude_2D: np.ndarray = magnitude.reshape(
            -1, magnitude.shape[2] * magnitude.shape[3]
        )
        phase_2D: np.ndarray = phase.reshape(-1, phase.shape[2] * phase.shape[3])

        # Apply SMOTE to the magnitude and phase of the FFT
        magnitude_2D, y = self.smote.fit_resample(magnitude_2D, y_2d)
        phase_2D, y = self.smote.fit_resample(phase_2D, y_2d)

        mag: np.ndarray = magnitude_2D.ravel()
        phas: np.ndarray = phase_2D.ravel()
        x: List[Any] = []
        for i in range(len(mag)):
            x.append(mag[i] * np.exp(1j * phas[i]))

        x = np.fft.ifft(x)
        magnitude = magnitude_2D.reshape(10, -1, x_train.shape[2], 641)
        phase = phase_2D.reshape(10, -1, x_train.shape[2], 641)
        y = y.reshape(10, -1, 5)

        # Return the processed data
        return x, y
