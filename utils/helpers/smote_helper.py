from imblearn.over_sampling import SMOTE
from typing import Tuple
import numpy as np


class SmoteHelper:
    """
    Helper class for Synthetic Minority Oversampling Technique (SMOTE).
    This balances the imbalance between 'rest' (re) and the other classes.
    More reading: https://imbalanced-learn.org/stable/over_sampling.html
    """

    @staticmethod
    def smote_processor(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to the input data.

        Parameters:
        x (np.ndarray): The input features.
        y (np.ndarray): The target values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The resampled features and target values.
        """
        sm: SMOTE = SMOTE(random_state=42)
        x_resampled: np.ndarray
        y_resampled: np.ndarray
        x_resampled, y_resampled = sm.fit_resample(x, y)
        return x_resampled, y_resampled
