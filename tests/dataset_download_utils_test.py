import os
import unittest
from unittest.mock import patch, MagicMock
from utils.dataset_download_utils import Downloader
import shutil
import zipfile
import tempfile
from io import BytesIO


class TestDownloader(unittest.TestCase):
    def setUp(self) -> None:
        """
        This method sets up the environment for each test.
        It initializes a Downloader instance with a URL and download directory.
        """
        self.url: str = "http://example.com/small_file.zip"
        self.download_dir: str = "test_dataset"
        self.download_path: str = os.path.join(self.download_dir, "small_file.zip")
        self.downloader: Downloader = Downloader(self.url, self.download_dir)

    def tearDown(self) -> None:
        """
        This method cleans up after each test.
        It removes the download directory and all its contents.
        """
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)

    @patch("requests.get")
    @patch("zipfile.is_zipfile", return_value=False)  # Mock not being a zip file
    def test_download(self, mock_zipcheck, mock_get: MagicMock) -> None:
        """
        This method tests the download functionality for non-zip files.
        It mocks the requests.get function and asserts that the file is downloaded.
        """
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.headers = {"content-length": "9"}
        mock_get.return_value = mock_response

        self.downloader.download()

        self.assertTrue(os.path.isfile(self.download_path))
        mock_zipcheck.assert_called_once_with(self.download_path)

    @patch("requests.get")
    @patch("zipfile.is_zipfile", return_value=True)  # Mock being a zip file
    def test_download_and_extract(self, mock_zipcheck, mock_get: MagicMock) -> None:
        """
        This method tests the download and extraction functionality for zip files.
        Mocks the requests.get function, asserts that the file is downloaded/extracted.
        """
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.headers = {"content-length": "9"}
        mock_get.return_value = mock_response

        zip_content = BytesIO()
        with zipfile.ZipFile(zip_content, "w") as zip_file:
            zip_file.writestr("file_inside_zip.txt", "Hello, world!")

        with patch("zipfile.ZipFile", return_value=zipfile.ZipFile(zip_content)):
            self.downloader.download()

        self.assertTrue(os.path.isfile(self.download_path))
        mock_zipcheck.assert_called_once_with(self.download_path)

    @patch("requests.get")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("os.path.isfile")
    def test_download_exception(self, mock_isfile, mock_open, mock_get):
        mock_isfile.return_value = False
        mock_get.side_effect = Exception("Test exception")
        with patch.object(self.downloader.logger, "error") as mock_error:
            self.downloader.download()
            mock_error.assert_called_once_with("An error occurred: Test exception")
