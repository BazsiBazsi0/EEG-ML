import unittest
import tempfile
import shutil
import os
import mne
from unittest.mock import Mock, patch
from helpers.filesystem_utils import FilesystemUtils


class TestFilesystemUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for testing
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory after testing
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Initialize the FilesystemUtils instance for each test
        self.fs_utils = FilesystemUtils()

    def tearDown(self):
        pass

    def test_list_files(self):
        # Capture the log messages
        with self.assertLogs(logger="helpers.filesystem_utils", level="INFO") as log:
            self.fs_utils.list_files()

        # Check if log messages contain the directory listing
        self.assertTrue(
            any(
                "Files and directories in a specified path:" in message
                for message in log.output
            )
        )

    def test_list_files2(self):
        with self.assertLogs(logger="filesystem_utils", level="INFO") as log:
            self.fs_utils.list_files()
            self.assertGreater(len(log.output), 0)

    def test_read_dir_files(self):
        # Create a dummy file and folder in the temporary directory
        test_file = os.path.join(self.test_dir, "test.txt")
        test_folder = os.path.join(self.test_dir, "test_folder")
        os.mkdir(test_folder)
        with open(test_file, "w") as f:
            f.write("Test")

        # Capture the log messages
        with self.assertLogs(logger=__name__, level="INFO") as log:
            self.fs_utils.read_dir_files()

        # Check if log messages contain the folder listing
        self.assertIn("Folders:", log.output)
        self.assertIn(test_folder, log.output)

    def test_dir_list(self):
        # Create dummy subdirectories in the temporary directory
        subdirs = ["dir1", "dir2", "dir3"]
        for subdir in subdirs:
            os.mkdir(os.path.join(self.test_dir, subdir))

        # Set the current working directory to the temporary directory
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        try:
            # Call the dir_list method and capture the result
            dir_list = self.fs_utils.dir_list()

            # Check if the result matches the expected list of subdirectories
            self.assertListEqual(dir_list, subdirs)
        finally:
            # Restore the original working directory
            os.chdir(original_cwd)

    @patch("mne.io.read_raw_edf", side_effect=Mock(return_value=None))
    def test_open_subj_files(self, mock_read_raw_edf):
        # Create a dummy subject directory
        subject_dir = os.path.join(self.test_dir, "subject")
        os.mkdir(subject_dir)

        # Create a dummy EDF file in the subject directory
        edf_file = os.path.join(subject_dir, "test.edf")
        with open(edf_file, "w") as f:
            f.write("EDF Test")

        # Call the open_subj_files method
        result = self.fs_utils.open_subj_files(subject_dir)

        # Check if the mock function was called
        mock_read_raw_edf.assert_called_once_with(edf_file)

        # Check if the result is as expected
        self.assertEqual(result, 0)
