import os
import requests
import zipfile
from utils.logging_utils import Logger
from tqdm import tqdm


class Downloader:
    def __init__(self, url: str, download_path: str = "dataset") -> None:
        self.url: str = url
        self.download_path: str = download_path
        self.logger: Logger = Logger(__name__)

    def download(self) -> None:
        filename: str = self.url.split("/")[-1]
        file_path: str = os.path.join(self.download_path, filename)

        # Create the directory if it doesn't exist
        os.makedirs(self.download_path, exist_ok=True)

        try:
            # Check if the file already exists
            if not os.path.isfile(file_path):
                self.logger.info(f"Downloading {self.url}...")
                response: requests.Response = requests.get(self.url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
                with open(file_path, "wb") as f:
                    for data in response.iter_content(chunk_size=1024):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                self.logger.info(f"Download completed. File saved to {file_path}")
            else:
                self.logger.info(f"File {file_path} already exists.")

            # If it's a zip file, extract it
            if zipfile.is_zipfile(file_path):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    if all(
                        [
                            os.path.exists(
                                os.path.join(self.download_path, member.filename)
                            )
                            for member in zip_ref.infolist()
                        ]
                    ):
                        self.logger.info(
                            f"All files in {file_path} are already extracted."
                        )
                    else:
                        self.logger.info(f"Extracting {file_path}...")
                        for member in tqdm(zip_ref.infolist(), desc="Extracting "):
                            zip_ref.extract(member, self.download_path)
                        self.logger.info(f"File {file_path} extracted.")
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
