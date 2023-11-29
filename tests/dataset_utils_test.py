import numpy as np
import mne
import unittest
import os
from unittest.mock import patch, MagicMock
from utils.dataset_utils import DatasetUtils
from utils.logging_utils import Logger


class TestDatasetUtils(unittest.TestCase):
    def setUp(self):
        self.dataset_utils = DatasetUtils()
        self.logger = Logger(__name__)

        # Filtering range for the test
        self.filtering = (0, 38)  # replace with your actual filtering range

        # Use an already existing file for testing
        # MNE cant create a raw EDF object from a non existing file
        self.path_run = "./dataset/files/S001/S001R01.edf"

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("numpy.save")
    def test_generate(self, mock_save, mock_exists, mock_makedirs):
        # Mock the os.makedirs function to do nothing
        mock_makedirs.return_value = None
        # Mock the os.path.exists function to always return True
        mock_exists.return_value = True
        # Mock the numpy.save function to do nothing
        mock_save.return_value = None

        # Call generate as it won't actually create any files
        try:
            self.dataset_utils.generate()
            self.logger.info("Generate method ran successfully")
        except Exception as e:
            self.logger.error(f"test_generate failed with {e}")
            self.fail(f"test_generate failed with {e}")

    def test_load_data(self):
        subject = 1
        data_path = self.dataset_utils.dataset_folder
        filtering = (1, 10)
        channel_level = 1
        channel_picks = ["C1..", "C2..", "C3..", "C4..", "C5..", "C6..", "Cz.."]

        xs, y = self.dataset_utils.load_data(
            subject, data_path, filtering, channel_level, channel_picks
        )

        self.assertIsInstance(xs, np.ndarray)
        self.assertIsInstance(y, list)

    def test_process_raw_edf(self):
        # Call the function with the mock EDF file and check the result
        result = self.dataset_utils.process_raw_edf(self.path_run, self.filtering)

        # Assert that the result is an instance of mne.io.Raw
        self.assertIsInstance(result, mne.io.BaseRaw)

    def test_label_epochs(self):
        # Create a dummy raw object with annotations
        info = mne.create_info(ch_names=10, sfreq=1000.0)
        raw = mne.io.RawArray(np.random.random((10, 1000)), info)
        raw.set_annotations(
            mne.Annotations(onset=[0.5], duration=[0.1], description=["T0"])
        )

        # Test for task2
        run = 4
        task2 = [4]
        task4 = []
        updated_raw = self.dataset_utils.label_epochs(raw.copy(), run, task2, task4)
        self.assertEqual(updated_raw.annotations.description[0], "B")

        # Test for task4
        run = 6
        task2 = []
        task4 = [6]
        updated_raw = self.dataset_utils.label_epochs(raw.copy(), run, task2, task4)
        self.assertEqual(updated_raw.annotations.description[0], "B")
