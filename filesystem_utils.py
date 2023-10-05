import os
import mne
from pathlib import Path
import logging
from typing import List, Union

# TODO validate + write a test for each function

# Create a custom logger
logger = logging.getLogger(__name__)


class FilesystemUtils:
    def __init__(self, debug: bool = False):
        self.debug = debug

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

        if self.debug:
            logger.info(dirlist)

        return dirlist

    def open_subj_files(self, subj_single: str) -> int:
        """
        Lists the files in a certain subject directory, filters *.edf files for safety.
        """
        source_dir: Union[Path, str] = Path(subj_single)
        filelist: List[str] = os.listdir(source_dir)

        for f in filelist:
            raw = mne.io.read_raw_edf(f)
            events, event_dict = mne.events_from_annotations(raw)
            event_dict = dict(rest=1, left=2, right=3)
            epochs = mne.Epochs(
                raw, events, event_id=event_dict, tmin=0, tmax=4, baseline=None
            )

        if self.debug:
            logger.info(filelist)

        return 0
