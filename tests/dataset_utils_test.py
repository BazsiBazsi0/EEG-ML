import unittest
from unittest.mock import patch
from utils.dataset_utils import DatasetUtils
from utils.logging_utils import Logger

class TestDatasetUtils(unittest.TestCase):
    def setUp(self):
        self.dataset_utils = DatasetUtils()
        self.logger = Logger(__name__)

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('numpy.save')
    def test_generate(self, mock_save, mock_exists, mock_makedirs):
        # Mock the os.makedirs function to do nothing
        mock_makedirs.return_value = None
        # Mock the os.path.exists function to always return True
        mock_exists.return_value = True
        # Mock the numpy.save function to do nothing
        mock_save.return_value = None

        # Now you can call generate and it won't actually create any files
        try:
            self.dataset_utils.generate()
            self.logger.info("generate method ran successfully")
        except Exception as e:
            self.logger.error(f"test_generate failed with {e}")
            self.fail(f"test_generate failed with {e}")