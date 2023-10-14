import os
import numpy as np
import mne
from typing import List, Tuple
from utils.config import excluded_pat
from utils.config import channel_inclusion_lvl
from utils.logging_utils import Logger
from utils.helpers.channel_picker_helper import ChannelPickerHelper
from utils.helpers.epoch_creator_helper import EpochCreatorHelper


class DatasetUtils:
    # This is based on the generator, aiming to be used for basic handling and filtering
    # of the dataset. It will create the real base dataset from the raw data.
    def __init__(
        self,
        dataset_folder: str = "dataset/files",
        subjects: list = [n for n in np.arange(0, 103) if n not in excluded_pat],
        channel_level: list = channel_inclusion_lvl,
        filtering: Tuple[int, int] = [0, 38],
    ):
        self.dataset_folder = dataset_folder
        self.subjects = subjects
        self.channel_level = channel_level
        self.filtering = filtering
        self.logger: Logger = Logger(__name__)

    def generate(self):
        """
        Generate and save filtered EEG data for multiple subjects and channel levels.

        This method performs the following operations:
        1. Constructs data and save paths based on provided parameters.
        2. Generates EEG data for different subjects and EEG channel levels.
        3. Saves the generated data to the specified save_path.

        Returns:
            None

        Note:
            This method iterates through the subjects, generates EEG data
            using the 'load_data' method, and saves the data in separate numpy files.

        """
        # Dir operations
        data_path = os.path.join(os.getcwd(), self.dataset_folder)

        # Generating the data, this is the part that does the processing
        # After loading x and y they are saved to the save_path into a numpy file
        for ch_level, ch_picks in self.channel_level.items():
            for sub in self.subjects:
                save_path = os.path.join(
                    os.getcwd(),
                    "dataset/filtered_data/ch_level_" + str(ch_level),
                )
                os.makedirs(save_path, exist_ok=True)

                x, y = DatasetUtils.load_data(
                    self,
                    subject=sub + 1,
                    data_path=data_path,
                    filtering=self.filtering,
                    channel_level=ch_level,
                    channel_picks=ch_picks,
                )

                np.save(os.path.join(save_path, "x_sub_" + str(sub)), x)
                np.save(os.path.join(save_path, "y_sub_" + str(sub)), y)

    def load_data(
        self,
        subject: int,
        data_path: str,
        filtering: Tuple[int, int],
        channel_level: int,
        channel_picks: list,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load data for a specific subject and return processed data and labels.

        Args:
            subject (int): The subject ID.
            data_path (str): Path to the data directory.
            filtering (List[int, int]): Filtering frequency range [low, high].
            channel_level (int): Channel level.
            channel_picks (list): Channel picks on a level.

        Returns:
            Tuple[np.ndarray, List[str]]: Processed data and corresponding labels.
        """
        # The imaginary runs for indexing purposes
        runs = [4, 6, 8, 10, 12, 14]
        # fists
        task2 = [4, 8, 12]
        # legs
        task4 = [6, 10, 14]
        # The subject naming scheme can be adapted using zero fill, example 'S001'
        sub_name = "S" + str(subject).zfill(3)
        # Generates a path for the folder of the subject
        sub_folder = os.path.join(data_path, sub_name)
        subject_runs = []

        for run in runs:
            path_run = os.path.join(
                sub_folder, sub_name + "R" + str(run).zfill(2) + ".edf"
            )
            raw_filt = self.process_raw_edf(path_run, filtering)
            epochs = self.label_epochs(raw_filt, run, task2, task4)
            subject_runs.append(epochs)

        xs, y = self.concat_and_return_data(subject_runs, channel_level, channel_picks)
        return xs, y

    def process_raw_edf(self, path_run: str, filtering: Tuple[int, int]) -> mne.io.Raw:
        """
        Process the raw EDF data file.

        Args:
            path_run (str): Path to the EDF data file.
            filtering (List[int, int]): Filtering frequency range [low, high].

        Returns:
            mne.io.Raw: Processed raw data.
        """
        # Read the raw edf file
        raw = mne.io.read_raw_edf(path_run, preload=True)
        # Filtering the data between 0 and 38 Hz
        raw_filt = raw.copy().filter(filtering[0], filtering[1])
        # This trims the run to 124 seconds precisely, default is set to 125 secs
        # 125 seconds * 160 Hz = 2000 data points
        # Necessary so we dont have overflowing data
        if np.sum(raw_filt.annotations.duration) > 124:
            raw_filt.crop(tmax=124)
        # Simple debugging feedback
        self.logger.info(
            f"Events from annotations: \n{mne.events_from_annotations(raw_filt)}"
        )
        self.logger.info(
            f"Raw original annotation: \n{raw_filt.annotations.description}"
        )
        return raw_filt

    def label_epochs(
        self, raw_filt: mne.io.Raw, run: int, task2: List[int], task4: List[int]
    ) -> mne.Epochs:
        """
        Label epochs based on task-specific annotations.

        Args:
            raw_filt (mne.io.Raw): Processed raw data.
            run (int): Current run number.
            task2 (List[int]): Run numbers for task2.
            task4 (List[int]): Run numbers for task4.

        Returns:
            mne.Epochs: Labeled epochs.

        Labeled annotations:
            - 'B' indicates baseline.
            - 'L' indicates motor imagination of opening and closing the left fist.
            - 'R' indicates motor imagination of opening and closing the right fist.
            - 'LR' indicates motor imagination of opening and closing both fists.
            - 'F' indicates motor imagination of moving both feet.

        The annotation description of the raw runs have <u2 (2 char unicode)
        dtype: {dtype[str_]:()} <U2.

        Description of the built-in data annotations:
            - The description for each run describes the sequence of
            - 'T0': rest
            - 'T1': motion real/imaginary
                - the left fist (in runs 3, 4, 7, 8, 11, and 12)
                - both fists (in runs 5, 6, 9, 10, 13, and 14)
            - 'T2': motion real/imaginary
                - the right fist (in runs 3, 4, 7, 8, 11, and 12)
                - both feet (in runs 5, 6, 9, 10, 13, and 14)

        If we print out the annotation descriptions,
        we would get 'T0' between all of the 'T1' and 'T2' annotations.
        It is easily recognizable that the meaning of 'T0-1-2'
        descriptions is dependent on the run numbers.
        """
        if run in task2:
            raw_filt.annotations.description = self.label_annotations(
                raw_filt.annotations.description, ["T0", "T1", "T2"], ["B", "L", "R"]
            )
        elif run in task4:
            raw_filt.annotations.description = self.label_annotations(
                raw_filt.annotations.description, ["T0", "T1", "T2"], ["B", "LR", "F"]
            )
        return raw_filt

    def label_annotations(
        self, descriptions: List[str], old_labels: List[str], new_labels: List[str]
    ) -> List[str]:
        """
        Label annotations based on a mapping of old labels to new labels.

        Args:
            descriptions (List[str]): List of annotations.
            old_labels (List[str]): List of old labels.
            new_labels (List[str]): List of new labels.

        Returns:
            List[str]: Updated annotations.
        """
        for i, desc in enumerate(descriptions):
            if desc in old_labels:
                descriptions[i] = new_labels[old_labels.index(desc)]
        return descriptions

    def concat_and_return_data(
        self,
        subject_runs: List[mne.Epochs],
        channel_level: list,
        channel_picks: list,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenate data from multiple runs and return processed data and labels.

        Args:
            subject_runs (List[mne.Epochs]): List of processed data from different runs.
            channel_level (int): Channel level.
            channel_picks (list): Channel picks on a level.

        Returns:
            Tuple[np.ndarray, List[str]]: Concatenated data and labels.
        """
        # Concatenate the runs
        raw_conc = mne.io.concatenate_raws(subject_runs, preload=True)
        # Assign a dummy "event_id" variable where we dump all the relevant data
        events, event_id = mne.events_from_annotations(raw_conc)
        # Renaming the dumped events using a standard dictionary
        event_id = {
            "rest": 1,
            "both_feet": 2,
            "left_hand": 3,
            "both_hands": 4,
            "right_hand": 5,
        }
        # Generating specific EEG epochs
        epochs = EpochCreatorHelper.create_epochs(raw_conc, events, event_id)
        # Picking the channels based on the channel level
        epochs = ChannelPickerHelper.pick_channels(
            epochs, channel_level, channel_picks, self.logger
        )
        # Constructing the data labels
        y = [epoch for epoch in epochs][:160]
        xs = np.array(epochs)[:160, :, :]
        return xs, y
