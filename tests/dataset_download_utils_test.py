import os
import unittest
from unittest.mock import patch, MagicMock
from tqdm import tqdm
from utils.dataset_download_utils import Downloader
import requests
import shutil
import zipfile
from io import BytesIO


class TestDownloader(unittest.TestCase):
    def setUp(self) -> None:
        self.url: str = "http://example.com/small_file.zip"
        self.download_dir: str = "test_dataset"
        self.download_path: str = os.path.join(self.download_dir, "small_file.zip")
        self.downloader: Downloader = Downloader(self.url, self.download_dir)

    def tearDown(self) -> None:
        # Clean up the created directory after each test
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)

    @patch("requests.get")
    @patch("zipfile.is_zipfile", return_value=False)  # Mock not being a zip file
    def test_download(self, mock_zipcheck, mock_get: MagicMock) -> None:
        # Mocking the response from requests.get
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.headers = {"content-length": "9"}
        mock_get.return_value = mock_response

        # Call the download method
        self.downloader.download()

        # Assert the file has been downloaded and written to disk
        self.assertTrue(os.path.isfile(self.download_path))
        # Assert that zipfile.is_zipfile was called
        mock_zipcheck.assert_called_once_with(self.download_path)

    @patch("requests.get")
    @patch("zipfile.is_zipfile", return_value=True)  # Mock being a zip file
    def test_download_and_extract(self, mock_zipcheck, mock_get: MagicMock) -> None:
        # Mocking the response from requests.get
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.headers = {"content-length": "9"}
        mock_get.return_value = mock_response

        # Create a temporary file-like object for the ZipFile content
        zip_content = BytesIO()
        with zipfile.ZipFile(zip_content, 'w') as zip_file:
            zip_file.writestr("file_inside_zip.txt", "Hello, world!")

        # Configure the mock ZipFile to return the temporary ZipFile content when opened
        with patch("zipfile.ZipFile", return_value=zipfile.ZipFile(zip_content)) as mock_zipfile:
            # Call the download method
            self.downloader.download()

        # Assert the file has been downloaded and written to disk
        self.assertTrue(os.path.isfile(self.download_path))
        # Assert that zipfile.is_zipfile was called
        mock_zipcheck.assert_called_once_with(self.download_path)
