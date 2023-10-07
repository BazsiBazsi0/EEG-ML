import unittest
import tempfile
import shutil
import os
from utils.filesystem_utils import FilesystemUtils


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
        with self.assertLogs(logger="utils.filesystem_utils", level="INFO") as log:
            self.fs_utils.list_files()

        # Check if log messages contain the directory listing
        self.assertTrue(
            any("Files and directories in a specified path:" in s for s in log.output)
        )

    def test_read_dir_files(self):
        # Create a dummy file and folder in the temporary directory
        test_file = os.path.join(self.test_dir, "test.txt")
        test_folder = os.path.join(self.test_dir, "test_folder")
        os.mkdir(test_folder)
        with open(test_file, "w") as f:
            f.write("Test")

        # Capture the log messages
        with self.assertLogs(logger="utils.filesystem_utils", level="INFO") as log:
            self.fs_utils.read_dir_files()

        # Check if log messages contain the folder listing
        self.assertTrue(any("Folders:" in s for s in log.output))

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
            self.assertListEqual(sorted(dir_list), sorted(subdirs))

        finally:
            # Restore the original working directory
            os.chdir(original_cwd)