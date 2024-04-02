import numpy as np
import utils.config as config
from typing import List
import os


class FileLoader:
    """
    A class that loads saved files from a specified dataset path.

    Methods:
        load_saved_files: Loads the saved files from the dataset path.

    Attributes:
        None
    """

    @staticmethod
    def load_saved_files(
        electrodes_load_level: int = 0,
        patient_id: int = None,
    ):
        """
        Loads the saved files from the dataset path.
        The output shape for x is (patients x epochs) x channels x datapoints. For a single patient,
        the shape would be (1 x 160) x 64 x 641, representing one patient with 160 epochs, 64 channels,
        and 641 datapoints. The output shape for y is (patients x epochs). For a single patient,
        the shape would be (1 x 160), representing one patient with 160 epochs, with a class label for each epoch.

        Args:
            electrodes_load_level (int): The level of electrode channel inclusion. Defaults to 0.
            patient_id (int): The ID of the patient to load. Defaults to None, which loads all patients.

        Returns:
            tuple: A tuple containing the loaded x and y data arrays.
        """
        # Set the paths for the dataset and filtered data
        dataset_path: str = config.dataset_path
        filtered_data_path: str = os.path.join(
            dataset_path, "filtered_data", f"ch_level_{electrodes_load_level}"
        )
        # Get a list of all x files in the filtered data path
        x_files: List[str] = [
            file for file in os.listdir(filtered_data_path) if file.startswith("x_")
        ]
        # Determine the number of patients based on the number of x files
        patients_full: int = len(x_files)
        # Create a list of subjects from 1 to the number of patients
        subjects: List[int] = list(range(1, patients_full + 1))
        # Get a list of excluded subjects from the config file
        excluded_subjects: List[int] = config.excluded
        # Remove the excluded subjects from the list of subjects
        subjects = set(set(subjects) - set(excluded_subjects))

        xs, ys = [], []
        for s in subjects:
            if patient_id is None or s == patient_id:
                xs.append(np.load(f"{filtered_data_path}/x_sub_{s}.npy"))
                ys.append(np.load(f"{filtered_data_path}/y_sub_{s}.npy"))

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        return x, y
