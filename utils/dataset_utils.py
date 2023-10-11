import os
import numpy as np
import mne
import config
import logging


class DatasetUtils:
    # This is based on the generator, aiming to be used for basic handling and filtering
    # of the dataset. It will create the real base dataset from the raw data.
    def __init__(
        self,
        dataset_folder: str = "dataset",
        subjects: list = [n for n in np.arange(0, 103) if n not in config.excluded_pat],
        ch_pick_level: list = [config.channel_inclusion_lvl[i] for i in range(3)],
        filtering: list = [0, 38],
    ):
        self.dataset_folder = dataset_folder
        self.subjects = subjects
        self.ch_pick_level = ch_pick_level
        self.filtering = filtering
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def generate(self):
        # Dir operations
        data_path = os.path.join(os.getcwd(), "dataset/files")
        save_path = os.path.join(
            os.getcwd(),
            "dataset/filtered_data/ch_level_" + str(self.ch_pick_level),
        )
        os.makedirs(save_path, exist_ok=True)

        # Generating the data, this is the part that does the processing
        # After loading x and y they are saved to the save_path into a numpy file
        for ch_level in self.ch_pick_level:
            for sub in self.subjects:
                x, y = DatasetUtils.load_data(sub, data_path, self.filtering, ch_level)

                np.save(os.path.join(save_path, "x_sub_" + str(sub)), x)
                np.save(os.path.join(save_path, "y_sub_" + str(sub)), y)

    @staticmethod
    def load_data(
        self, subject: int, data_path: str, filtering: [int, int], ch_pick_level: int
    ) -> (np.ndarray, list, list):
        # The imaginary runs for indexing purposes
        runs = [4, 6, 8, 10, 12, 14]
        # fists
        task2 = [4, 8, 12]
        # legs
        task4 = [6, 10, 14]

        # The subject naming scheme can be adapted using zero fill(z-fill), example 'S001'
        sub_name = "S" + str(subject).zfill(3)

        # Generates a path for the folder of the subject
        sub_folder = os.path.join(data_path, sub_name)
        subject_runs = []

        # Processing each run individually for each subject
        for run in runs:
            # Generate the path using the folder path, with specifying the run
            path_run = os.path.join(
                sub_folder, sub_name + "R" + str(run).zfill(2) + ".edf"
            )

            # Read the raw edf file
            raw = mne.io.read_raw_edf(path_run, preload=True)

            # Filtering the data between 0 and 38 Hz
            raw_filt = raw.copy().filter(filtering[0], filtering[1])

            # This trims the run to 124 seconds precisely, default is set to 125 secs
            # 125 seconds * 160 Hz = 2000 data points
            # Necessary so we dont have overflowing data
            if np.sum(raw_filt.annotations.duration) > 124:
                raw_filt.crop(tmax=124)

            # Now we need to label epochs based on the annotations
            # Simple debugging feedback
            self.logger.info(
                f"Events from annotations: {mne.events_from_annotations(raw_filt)}"
            )
            self.logger.info(
                f"Raw annotation original descriptions: \n{raw_filt.annotations.description}"
            )

            # if-for block with the previously defined arrays for runs
            if run in task2:
                for index, annotation in enumerate(raw_filt.annotations.description):
                    if annotation == "T0":
                        raw_filt.annotations.description[index] = "B"
                    if annotation == "T1":
                        raw_filt.annotations.description[index] = "L"
                    if annotation == "T2":
                        raw_filt.annotations.description[index] = "R"
            if run in task4:
                for index, annotation in enumerate(raw_filt.annotations.description):
                    if annotation == "T0":
                        raw_filt.annotations.description[index] = "B"
                    if annotation == "T1":
                        raw_filt.annotations.description[index] = "LR"
                    if annotation == "T2":
                        raw_filt.annotations.description[index] = "F"
            self.logger.info(
                f"Raw annotation original descriptions: \n\{raw_filt.annotations.description}"
            )

            subject_runs.append(raw_filt)

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

        # Excluding bad channels and any other data that could get in for safety
        picks = mne.pick_types(
            raw_conc.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        # Generating epochs
        epochs = mne.epochs.Epochs(
            raw_conc,
            events,
            event_id,
            tmin=0,
            tmax=4,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        # Sanity check:
        self.logger.info("EEG channels are being selected.")

        # Selecting channels
        # Selection lists:

        channel_pick_lvl: list = [
            config.channel_inclusion_lvl[0],
            config.channel_inclusion_lvl[1],
            config.channel_inclusion_lvl[2],
        ]

        self.logger.info(f"EEG channels before selection: \n{epochs[0].ch_names}")
        # select channels that are only above the central sulcus
        if ch_pick_level == 0:
            epochs.pick_channels(ch_names=channel_pick_lvl[0])
        elif ch_pick_level == 1:
            epochs.pick_channels(ch_names=channel_pick_lvl[1])
        elif ch_pick_level == 2:
            epochs.pick_channels(ch_names=channel_pick_lvl[2])
        elif ch_pick_level == 3:
            pass
        else:
            self.logger.info("EEG channel selection level is not defined")
            return

        if len(epochs.ch_names) == len(channel_pick_lvl[ch_pick_level]):
            print("Channel selection successful.")
        else:
            raise ValueError("Channel selection failed.")

        self.logger.info(f"EEG channels remaining after selection: \n{epochs[0].ch_names}")
        
        # Constructing the data labels
        y = list()
        for index, data in enumerate(epochs):
            y.append(epochs[index]._name)

        # Retuning with exactly 4 seconds epochs in both x and y
        xs = np.array(epochs)
        xs = xs[:160, :, :]
        return xs[:160, :, :], y[:160]
