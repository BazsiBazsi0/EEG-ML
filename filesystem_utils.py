import os
import mne
from pathlib import Path


class FilesystemUtils:

    @staticmethod
    def list_files(self):
        # This only lists files in cwd
        # prints the files found
        # maybe use for sanity check later
        path_of_the_directory = os.getcwd()
        # path_of_the_directory= 'EEG_Physionet'
        print("Files and directories in a specified path:")
        for filename in os.listdir(path_of_the_directory):
            f = os.path.join(path_of_the_directory, filename)
            if os.path.isfile(f):
                print(f)

    @staticmethod
    def read_dir_files(self):
        # this reads in folders and files
        # python 3.5 style with scandir
        # the files & dirs in the root(current working dir) are saved in arrays
        # prints the dirs found, but the object are not simple strings
        # I can't make use of it currently

        folders = []
        folder_paths = []
        files = []
        for entry in os.scandir(os.getcwd()):
            if entry.is_dir():
                folders.append(entry)
                folder_paths.append(entry.path)
            elif entry.is_file():
                files.append(entry.path)
        print('Folders:')
        for f in folders:
            print(f)

    @staticmethod
    def dir_list(self):
        # this takes the list of dirs and put them into a list
        dirlist = [item for item in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), item))]

        if self.debug:
            print(dirlist)

        return dirlist

    @staticmethod
    def open_subj_files(self, subj_single):
        # This lists the files in certain subject dir, filters *.edf files for safety
        source_dir = Path(subj_single)
        filelist = os.listdir(source_dir)

        for f in filelist:
            raw = mne.io.read_raw_edf(f)
            events, event_dict = mne.events_from_annotations(raw)
            event_dict = dict(rest=1, left=2, right=3)
            epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=0, tmax=4, baseline=None)

        # below is file processing if needed in any case
        """
        files = source_dir.glob('*.edf')
        for file in files:
            with file.open('r') as file_handle:
                for single_file in file_handle:
                    # do your thing
                    yield single_file
        """

        if self.debug:
            print(filelist)

        return 0
