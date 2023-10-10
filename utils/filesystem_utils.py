import os
import logging
from typing import List

# TODO validate + write a test for each function

# Create a custom logger
logger = logging.getLogger(__name__)


class FilesystemUtils:
    def list_files(self) -> None:
        """
        Lists files in the current working directory.
        """
        path_of_the_directory: str = os.getcwd()
        logger.info("Files and directories in a specified path:")
        for filename in os.listdir(path_of_the_directory):
            f: str = os.path.join(path_of_the_directory, filename)
            if os.path.isfile(f):
                logger.info(f)

    def read_dir_files(self) -> None:
        """
        Reads in folders and files in the current working directory.
        """
        folders: List[str] = []
        folder_paths: List[str] = []
        files: List[str] = []
        for entry in os.scandir(os.getcwd()):
            if entry.is_dir():
                folders.append(entry)
                folder_paths.append(entry.path)
            elif entry.is_file():
                files.append(entry.path)
        logger.info("Folders:")
        for f in folders:
            logger.info(f)

    def dir_list(self) -> List[str]:
        """
        Returns a list of directories in the current working directory.
        """
        dirlist: List[str] = [
            item
            for item in os.listdir(os.getcwd())
            if os.path.isdir(os.path.join(os.getcwd(), item))
        ]

        logger.info(dirlist)

        return dirlist
