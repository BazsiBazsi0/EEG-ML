import unittest
from unittest import mock
import numpy as np
from fileloader import FileLoader


class TestFileLoader(unittest.TestCase):
    @mock.patch("os.listdir")
    @mock.patch("numpy.load")
    @mock.patch("fileloader.config")
    def test_load_saved_files(self, mock_config, mock_np_load, mock_listdir):
        # Mocking the config attributes
        mock_config.dataset_path = "/fake/path"
        mock_config.excluded = [2]

        # Mocking os.listdir to return a list of files
        mock_listdir.return_value = ["x_sub_1.npy", "x_sub_2.npy", "x_sub_3.npy"]

        # Mocking numpy.load to return a numpy array
        mock_np_load.return_value = np.array([1, 2, 3])

        # Call the method to test
        x, y = FileLoader.load_saved_files()

        # Assert the numpy.load was called with the correct arguments
        mock_np_load.assert_any_call("/fake/path/filtered_data/ch_level_0/x_sub_1.npy")
        mock_np_load.assert_any_call("/fake/path/filtered_data/ch_level_0/x_sub_3.npy")
        mock_np_load.assert_any_call("/fake/path/filtered_data/ch_level_0/y_sub_1.npy")
        mock_np_load.assert_any_call("/fake/path/filtered_data/ch_level_0/y_sub_3.npy")

        # Assert the returned x and y are as expected
        self.assertTrue((x == np.array([1, 2, 3, 1, 2, 3])).all())
        self.assertTrue((y == np.array([1, 2, 3, 1, 2, 3])).all())
