import os
import numpy as np
import mne
import pickle


# This class generates files to be used for loading later on
class Generator:

    @staticmethod
    def process_subject_data(subject: int, data_path: str):
        """
        Takes a single subject and preprocesses it for the generator function
        The data path is given a simple string
        The function returns:
            xs: Time series of the data with a shape of(n_sample, 64, 641)
            y: The list of labels with a length of n_samples
            ch_names: 64 channels in the xs array
        """

        # The imaginary runs for indexing purposes
        runs = [4, 6, 8, 10, 12, 14]
        task2 = [4, 8, 12]
        task4 = [6, 10, 14]

        # The subject naming scheme can be adapted using zero fill(z-fill)
        sub_name = 'S' + str(subject).zfill(3)

        # Generates a path for the folder of the subject
        sub_folder = os.path.join(data_path, sub_name)
        subject_runs = []

        # Processing each run individually for each subject
        for run in runs:
            # Here I also use zero-fill, generates the path using the folder path, with specifying the run
            path_run = os.path.join(sub_folder, sub_name + 'R' + str(run).zfill(2) + '.edf')

            # Finally reading the raw edf file
            raw_filt = mne.io.read_raw_edf(path_run, preload=True)
            # Checking the file for a single run
            # raw_run.plot_psd(fmax=80);

            raw_filt = raw_filt.copy().filter(0.1, 30)
            # raw_filt.plot_psd(fmax=80);

            # This trims the run to 124 sec precisely because by default its set to 125 secs
            # 125 seconds * 160 Hz = 2000 data points

            if np.sum(raw_filt.annotations.duration) > 124:
                raw_filt.crop(tmax=124)

            # The result is a somewhat smaller run in terms of length, currently im not sure why this is needed
            # Now we need to make epochs based on the annotations
            """
            B indicates baseline
            L indicates motor imagination of opening and closing left fist;
            R indicates motor imagination of opening and closing right fist;
            LR indicates motor imagination of opening and closing both fists;
            F indicates motor imagination of opening and closing both feet.
            I want to figure out why the indexes are only accepting two chars
            Looks like the annotation description of the raw runs have <u2(2 char unicode) dtype: {dtype[str_]:()} <U2
            Very upsetting since we dont get any errors if we want to assign a longer string to them
            """

            """
            The description for each run describes the sequence of
                T0: rest
                T1: motion real/imaginary
                    the left fist (in runs 3, 4, 7, 8, 11, and 12)
                    both fists (in runs 5, 6, 9, 10, 13, and 14)
                T2: motion real/imaginary
                    the right fist (in runs 3, 4, 7, 8, 11, and 12)
                    both feet (in runs 5, 6, 9, 10, 13, and 14)
            So if we print out the annotation descriptions we would get T0 between all of the T1 and T2 annotations.
            It is easily recognisable that the meaning of 'T0-1-2' descriptions are dependent on the run numbers.
            """
            print("Events from annotations: ", mne.events_from_annotations(raw_filt))
            print('Raw annotation original descriptions: \n', raw_filt.annotations.description)
            if int(run) in task2:
                for index, annotation in enumerate(raw_filt.annotations.description):
                    if annotation == "T0":
                        raw_filt.annotations.description[index] = 'B'
                    if annotation == "T1":
                        raw_filt.annotations.description[index] = "L"
                    if annotation == "T2":
                        raw_filt.annotations.description[index] = "R"
            if int(run) in task4:
                for index, annotation in enumerate(raw_filt.annotations.description):
                    if annotation == "T0":
                        raw_filt.annotations.description[index] = "B"
                    if annotation == "T1":
                        raw_filt.annotations.description[index] = "LR"
                    if annotation == "T2":
                        raw_filt.annotations.description[index] = "F"
            print('Raw annotation modified descriptions: \n', raw_filt.annotations.description)
            subject_runs.append(raw_filt)
        # After re-classifying each run into their own category the annotations are proprely labeled
        # print(subject_runs[0].annotations.description)
        raw_conc = mne.io.concatenate_raws(subject_runs, preload=True)

        # This might be redundant since we are already excluding bads at the epoch generation below
        """indexes = []
        for index, value in enumerate(raw_conc.annotations.description):
            if value == "BAD boundary" or value == "EDGE boundary":
                indexes.append(index)
        raw_conc.annotations.delete(indexes)"""

        # Standardizes ch position and names
        mne.datasets.eegbci.standardize(raw_conc)

        # Montage: a common word used to describe the arrangement of EEG electrodes on a head, or in an EEG cap.
        # Here the standard 10-5 system is used in the original code but it's wrong(343+3 locations)
        # There are only 64 channels
        # Physionet states that they are using 10-10 system(65+3 electrodes) and the following electrodes are excluded:
        #    Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10
        # This is not really needed in our case
        # raw_conc.set_montage(mne.channels.make_standard_montage('standard_1005'))

        # I can't plot the sensor positions somehow, all the plots are returning black, maybe old hw issue
        # raw_conc.plot_sensors()

        # First i assign a dummy event_id variable where I dump all the relevant data about our epoch, then rename them
        events, event_id = mne.events_from_annotations(raw_conc)

        # Renaming the events using a standard dictionary
        # event_id = dict(rest=1, both_feet=2, left=3, left_right_hands=4, right=5)

        event_id = {
            'rest': 1,
            'both_feet': 2,
            'left_hand': 3,
            'both_hands': 4,
            'right_hand': 5
        }
        # Excluding bad channels
        picks = mne.pick_types(raw_conc.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        # Generating epochs, SSP is enabled, as an option here we could reject bad boundaries if needed
        epochs = mne.epochs.Epochs(raw_conc, events, event_id, tmin=0, tmax=4, proj=True, picks=picks, baseline=None,
                                   preload=True)

        print('EEG channels: \n', epochs[0].ch_names)

        y = list()
        for index, data in enumerate(epochs):
            y.append(epochs[index]._name)

        # This list comprehension does the same as casting into an np-array
        # xs = np.array([epoch for epoch in epochs])
        xs = np.array(epochs)
        return xs, y, raw_conc.ch_names

    @staticmethod
    def generate():
        # Exclusion for subjects due to annotation errors
        exclude = [38, 88, 89, 92, 100, 104]
        # List comprehension for generating a list without excluded subjects
        subjects = [n for n in np.arange(1, 110) if n not in exclude]
        # Save path: needs to be created before running
        save_path = "dataset/sub_by_sub_motor_imagery"
        # Goes through subjects and dumps the processed x, y , ch data into a file for each subjects
        for subject in subjects:
            x, y, ch_names = Generator.process_subject_data(subject, "raw_data")
            # Creates files with write-binary using pickleing
            with open(os.path.join(save_path, str(subject) + ".pkl"), "wb") as file:
                pickle.dump([x, y, ch_names], file)
