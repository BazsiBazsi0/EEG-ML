import os
import unittest
from unittest.mock import patch, MagicMock
from tqdm import tqdm
from utils.dataset_download_utils import Downloader
import requests
import shutil


class TestDownloader(unittest.TestCase):
    def setUp(self) -> None:
        self.url: str = (
            "http://example.com/small_file.zip"  # Smaller file for faster download
        )
        self.filename: str = self.url.split("/")[-1]
        self.download_dir: str = "test_dataset"
        self.download_path: str = os.path.join(self.download_dir, self.filename)
        # Use the correct download directory in the Downloader instance
        self.downloader: Downloader = Downloader(self.url, self.download_path, self.download_dir)

    def tearDown(self) -> None:
        # Clean up the created directory after each test
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)

    @patch("requests.get")
    def test_download(self, mock_get: MagicMock) -> None:
        # Mocking the response from requests.get
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.headers = {"content-length": "9"}
        mock_get.return_value = mock_response

        # Download with progress bar
        response = requests.get(self.url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        with open(self.download_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

        # Assert the file has been downloaded and written to disk
        self.assertTrue(os.path.isfile(self.download_path))