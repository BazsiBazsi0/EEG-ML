import os
import numpy as np
import mne


# This class generates files to be used for loading later on
class Generator:
    # Using static method the function should not be dependent on initialization
    # Because i just want to use them current as a self-contained functions
    # TODO redundant lines around enumerations
    @staticmethod
    def load_data(
        subject: int, data_path: str, filtering: [int, int], ch_pick_level: int
    ) -> (np.ndarray, list, list):
        """
        For more info for return types check https://docs.python.org/3/library/typing.html
        The BasedRaw format: https://mne.tools/stable/generated/mne.io.BaseRaw.html?highlight=baseraw#mne.io.BaseRaw
        Source is partially from https://github.com/Kubasinska/MI-EEG-1D-CNN processor
        Description of the function:
        Takes a single subject and preprocesses it for the generator function
        The data path is given a simple string
        The function returns:
            xs: Time series ndarray of the data with a shape of(n_sample, 64, 641)
            y: The list of labels with a length of n_samples
            ch_names: The 64 channels in the xs array as str list
        """

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
            # Here I also use zero-fill, generates the path using the folder path, with specifying the run
            path_run = os.path.join(
                sub_folder, sub_name + "R" + str(run).zfill(2) + ".edf"
            )

            # Finally reading the raw edf file
            raw = mne.io.read_raw_edf(path_run, preload=True)
            # Sanity check: Checking the file for a single run
            # raw_run.plot_psd(fmax=80);

            # Filtering the data between 0 and 38 Hz
            raw_filt = raw.copy().filter(filtering[0], filtering[1])

            # Sanity check: Checking the file for a single run after filtering
            # raw_filt.plot_psd(fmax=80);

            # This trims the run to 124 seconds precisely, default is set to 125 secs
            # 125 seconds * 160 Hz = 2000 data points
            # We need this so we dont have overflowing data
            if np.sum(raw_filt.annotations.duration) > 124:
                raw_filt.crop(tmax=124)

            # Now we need to label epochs based on the annotations
            """
            B indicates baseline
            L indicates motor imagination of opening and closing left fist;
            R indicates motor imagination of opening and closing right fist;
            LR indicates motor imagination of opening and closing both fists;
            F indicates motor imagination of moving both feet.
            
            The annotation description of the raw runs have <u2(2 char unicode) dtype: {dtype[str_]:()} <U2
            """
            # Description of the built-in data annotations
            """
            The description for each run describes the sequence of
                T0: rest
                T1: motion real/imaginary
                    the left fist (in runs 3, 4, 7, 8, 11, and 12)
                    both fists (in runs 5, 6, 9, 10, 13, and 14)
                T2: motion real/imaginary
                    the right fist (in runs 3, 4, 7, 8, 11, and 12)
                    both feet (in runs 5, 6, 9, 10, 13, and 14)
            If we print out the annotation descriptions we would get T0 between all of the T1 and T2 annotations.
            It is easily recognisable that the meaning of 'T0-1-2' descriptions are dependent on the run numbers.
            """
            # Simple debugging feedback
            print("Events from annotations: ", mne.events_from_annotations(raw_filt))
            print(
                "Raw annotation original descriptions: \n",
                raw_filt.annotations.description,
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
            print(
                "Raw annotation modified descriptions: \n",
                raw_filt.annotations.description,
            )
            subject_runs.append(raw_filt)

        # After re-classifying each run into their own category the annotations are properly labeled
        # Sanity check:
        # print(subject_runs[0].annotations.description)

        # Concatenate the runs
        raw_conc = mne.io.concatenate_raws(subject_runs, preload=True)

        # First I assign a dummy "event_id" variable where I dump all the relevant data about our epoch, then rename them
        events, event_id = mne.events_from_annotations(raw_conc)

        # Renaming the events using a standard dictionary
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
        print("EEG channels are being selected.")

        # Selecting channels
        # Selection lists:

        # Selection lists:
        channel_pick_lvl = [
            ["C1..", "C2..", "C3..", "C4..", "C5..", "C6..", "Cz.."],
            [
                "C1..",
                "C2..",
                "C3..",
                "C4..",
                "C5..",
                "C6..",
                "Cz..",
                "Fc1.",
                "Fc2.",
                "Fc3.",
                "Fc4.",
                "Fc5.",
                "Fc6.",
                "Fcz.",
                "Cp1.",
                "Cp2.",
                "Cp3.",
                "Cp4.",
                "Cp5.",
                "Cp6.",
                "Cpz.",
            ],
            [
                "C1..",
                "C2..",
                "C3..",
                "C4..",
                "C5..",
                "C6..",
                "Cz..",
                "Fc1.",
                "Fc2.",
                "Fc3.",
                "Fc4.",
                "Fc5.",
                "Fc6.",
                "Fcz.",
                "Cp1.",
                "Cp2.",
                "Cp3.",
                "Cp4.",
                "Cp5.",
                "Cp6.",
                "Cpz.",
                "F1..",
                "F2..",
                "F3..",
                "F4..",
                "F5..",
                "F6..",
                "Fz..",
                "Af3.",
                "Af4.",
                "Afz.",
                "P1..",
                "P2..",
                "P3..",
                "P4..",
                "P5..",
                "P6..",
                "Pz..",
                "Po3.",
                "Po4.",
                "Poz.",
            ],
        ]

        print("EEG channels before selection: \n", epochs[0].ch_names)
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
            print("EEG channel selection level is not defined")
            return

        if len(epochs.ch_names) == len(channel_pick_lvl[ch_pick_level]):
            print("Channel selection successful.")
        else:
            raise ValueError("Channel selection failed.")

        print("EEG channels remaining after selection: \n", epochs[0].ch_names)
        # Construting the data labels
        y = list()
        for index, data in enumerate(epochs):
            y.append(epochs[index]._name)

        # TODO: Separate data management """epochs.get_data..."""
        #   epochs all in one query
        # TODO check out epochs.equalize_event_counts()

        # Returing with exactly 4 seconds epochs in both x and y
        xs = np.array(epochs)
        xs = xs[:160, :, :]
        return xs[:160, :, :], y[:160]

    @staticmethod
    def generate():
        """
        This method generates the data for the model
        Running time should be pretty fast unless you run it on a toaster
        Memory should be safe but if you have a toaster you might want to monitor that
        """

        # Filtering the subjects
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(0, 103) if n not in exclude]

        # Picking the filtering level for the channels
        ch_pick_level = 1

        # Filtering for the frequencies
        filtering = [0, 38]

        # Dir operations
        data_path = os.path.join(os.getcwd(), "raw_data")
        save_path = os.path.join(
            os.getcwd(),
            "generator/all_electrodes_103_patients_ch_level_" + str(ch_pick_level),
        )
        os.makedirs(save_path, exist_ok=True)

        # Generating the data, this is the part that does the processing
        # After loading x and y they are saved to the save_path into a numpy file
        for sub in subjects:
            x, y = Generator.load_data(103, data_path, filtering, ch_pick_level)

            np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
            np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
