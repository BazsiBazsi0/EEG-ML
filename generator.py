import os
import numpy as np
import mne
import pickle

# This class generates files to be used for loading later on
class Generator:

    # using static method the function should not be dependent on initialization
    # because i just want to use them current as a self-contained functions
    # TODO redundant lines around enumerations
    @staticmethod
    def load_data(subjects: list, runs: list, data_path: str) -> (np.ndarray, list):
        # this introduces typing module for typing hints, type assignments. Check mypy for more info and analysis.
        # for more info check https://docs.python.org/3/library/typing.html and http://mypy-lang.org (its epic)
        # The BasedRaw format: https://mne.tools/stable/generated/mne.io.BaseRaw.html?highlight=baseraw#mne.io.BaseRaw
        # Base class for Raw data.
        # Source from https://github.com/Kubasinska/MI-EEG-1D-CNN processor
        # Take in subject list, runs list, data path as string, and returns a list consisting of lists of BaseRaw
        # Iterates through each folder, and each run loads them and modifies the labels, then returns a list

        # In the end it returns the final epochs and a list
        # TODO Replace the if statements with something, possibly with one of the functions written
        all_subject_list = []
        data_path = 'raw_data'
        subjects = [str(s) for s in subjects]
        runs = [str(r) for r in runs]
        # selecting imaginary tasks, we only need these to train the network
        task2 = [4, 8, 12]  # fists
        task4 = [6, 10, 14]  # legs
        # iterating through the subjects list(a simple str list)
        for sub in subjects:
            # naming scheme for the folders
            # TODO there is a function that fills the name with 0s might be use ful for sub_name-ing scheme
            # use 0-paddig f-string
            sub_name = 'S'+sub.zfill(3)
            sub_folder = os.path.join(data_path, sub_name)
            single_subject_run = []

            for run in runs:
                path_run = os.path.join(sub_folder, sub_name + 'R'+run.zfill(2)+'.edf')

                raw_run = mne.io.read_raw_edf(path_run, preload=True)
                len_run = np.sum(raw_run._annotations.duration)
                if len_run > 124:
                    print(sub)
                    raw_run.crop(tmax=124)

                """
                rest(baseline)
                left: motor imagination of opening and closing left fist;
                right:motor imagination of opening and closing right fist;
                both_fist: indicates motor imagination of opening and closing both fists;
                both_feet: indicates motor imagination of opening and closing both feet.
                """
                # TODO replace this with normal looking list/dict styled structure
                if int(run) in task2:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "re"   # Rest
                        if an == "T1":
                            raw_run.annotations.description[index] = "le"   # Left
                        if an == "T2":
                            raw_run.annotations.description[index] = "ri"   # Right
                if int(run) in task4:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "re"   # Rest
                        if an == "T1":
                            raw_run.annotations.description[index] = "fi"   # Fists
                        if an == "T2":
                            raw_run.annotations.description[index] = "fe"   # Feet
                single_subject_run.append(raw_run)
            all_subject_list.append(single_subject_run)
        # returns with type list[list[mne.io.BaseRaw]], i commented it out beacuse we need to concatenate and del annot
        ### return all_subject_list

        # concatenating runs
        raw_conc_list = []
        for subj in all_subject_list:
            raw_conc = mne.io.concatenate_raws(subj)
            raw_conc_list.append(raw_conc)
        # return type: List[BaseRaw],  still processing afterwards
        ### return raw_conc_list

        # deleting edge and bad boundary annotations from raws
        raw_bound_cleaned_list = []
        for subj in raw_conc_list:
            indexes = []
            for index, value in enumerate(subj.annotations.description):
                if value == "BAD boundary" or value == "EDGE boundary":
                    indexes.append(index)
            subj.annotations.delete(indexes)
            # cleaned list
            raw_bound_cleaned_list.append(subj)

        # return type: List[BaseRaw], still processing
        ### return raw_bound_cleaned_list

        raw_standardized = []
        for subj in raw_bound_cleaned_list:
            mne.datasets.eegbci.standardize(subj)
            montage = mne.channels.make_standard_montage('standard_1005')
            subj.set_montage(montage)
            raw_standardized.append(subj)

        # TODO channel picking/sorting

        subj_list_with_ch = []
        for raw_std in raw_standardized:
            subj_list_with_ch.append(raw_std.pick_channels(["FC1", "FC2"]))
        # now we need to split up the data into numpy epochs
        # from t=0 to t=4

        # TODO baseline needed?

        # epoch creator
        # TODO investigate better x and y

        xs = list()
        ys = list()
        tmin: int = 0
        tmax: int = 4
        event_id = dict(rest=1, both_feet=2, left=3, both_fist=4, right=5)
        event_dict = []
        for r in subj_list_with_ch:
            events, event_dict = mne.events_from_annotations(r)
            # print(event_dict)
            # events, _ = mne.events_from_annotations(r)
            picks = mne.pick_types(r.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            epochs = mne.epochs.Epochs(r, events, event_dict, tmin, tmax, proj=True, picks=picks, baseline=None,
                                       preload=True)

        y = list()
        for index, data in enumerate(epochs):
            y.append(epochs[index]._name)

        xs.append(np.array([epoch for epoch in epochs]))
        ys.append(y)
        # returns a concatenated xs, and a ys which is a pretty rough list comprehension
        # for sublist in ys:
        #    for item in sublist:
        #       appends the item
        return np.concatenate(tuple(xs), axis=0), [item for sublist in ys for item in sublist]


    @staticmethod
    def generate():
        exclude = [38, 88, 89, 92, 100, 104]
        # smartypants list comprehension for making a list with exclusion
        subjects = [n for n in np.arange(1, 10) if n not in exclude]
        # subjects = [1]
        runs = [4, 6, 8, 10, 12, 14]
        data_path = os.getcwd()
        save_path = os.path.join(os.getcwd(), "test_generate/second_gen")
        # os.makedirs(save_path)
        for sub in subjects:
            x, y = Generator.load_data(subjects, runs, data_path)

            np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
            np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)

    @staticmethod
    def load_subject_data(subject: int, data_path: str, exclude_base: bool = False):
        """
        Given a subject number (@subject) and the original dataset
        path (@data_path), this function returns:
            xs: The time series; a numpy array of shape (n_sample, 64, 641)
            y: The labels, a list of length n_samples
            ch_names: The 64 channels order in the xs array
        """
        runs = [4, 6, 8, 10, 12, 14]
        task2 = [4, 8, 12]
        task4 = [6, 10, 14]
        if len(str(subject)) == 1:
            sub_name = "S" + "00" + str(subject)
        elif len(str(subject)) == 2:
            sub_name = "S" + "0" + str(subject)
        else:
            sub_name = "S" + str(subject)
        sub_folder = os.path.join(data_path, sub_name)
        subject_runs = []
        for run in runs:
            if len(str(run)) == 1:
                path_run = os.path.join(sub_folder,
                                        sub_name + "R" + "0" + str(run) + ".edf")
            else:
                path_run = os.path.join(sub_folder,
                                        sub_name + "R" + str(run) + ".edf")
            raw_run = mne.io.read_raw_edf(path_run, preload=True)
            len_run = np.sum(
                raw_run._annotations.duration)
            if len_run > 124:
                raw_run.crop(tmax=124)

            """
            B indicates baseline
            L indicates motor imagination of opening and closing left fist;
            R indicates motor imagination of opening and closing right fist;
            LR indicates motor imagination of opening and closing both fists;
            F indicates motor imagination of opening and closing both feet.
            """

            if int(run) in task2:
                for index, an in enumerate(raw_run.annotations.description):
                    if an == "T0":
                        raw_run.annotations.description[index] = "B"
                    if an == "T1":
                        raw_run.annotations.description[index] = "L"
                    if an == "T2":
                        raw_run.annotations.description[index] = "R"
            if int(run) in task4:
                for index, an in enumerate(raw_run.annotations.description):
                    if an == "T0":
                        raw_run.annotations.description[index] = "B"
                    if an == "T1":
                        raw_run.annotations.description[index] = "LR"
                    if an == "T2":
                        raw_run.annotations.description[index] = "F"
            subject_runs.append(raw_run)
        raw_conc = mne.io.concatenate_raws(subject_runs)
        indexes = []
        for index, value in enumerate(raw_conc.annotations.description):
            if value == "BAD boundary" or value == "EDGE boundary":
                indexes.append(index)
        raw_conc.annotations.delete(indexes)

        mne.datasets.egbci.standardize(raw_conc)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw_conc.set_montage(montage)
        tmin = 0
        tmax = 4
        if exclude_base:
            event_id = dict(F=2, L=3, LR=4, R=5)
        else:
            event_id = dict(B=1, F=2, L=3, LR=4, R=5)

        events, _ = mne.events_from_annotations(raw_conc, event_id=event_id)

        picks = mne.pick_types(raw_conc.info, meg=False, eeg=True, stim=False,
                               eog=False, exclude='bads')
        epochs = mne.epochs.Epochs(raw_conc, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)

        print(epochs[0].ch_names)

        y = list()
        for index, data in enumerate(epochs):
            y.append(epochs[index]._name)

        xs = np.array([epoch for epoch in epochs])

        return xs, y, raw_conc.ch_names

    @staticmethod
    def generate_new():
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 110) if n not in exclude]
        save_path = "dataset/sub_by_sub_motor_imagery"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for subject in subjects:
            x, y, ch_names = Generator.load_subject_data(subject, "raw_data")
            with open(os.path.join(save_path, str(subject) + ".pkl"), "wb") as file:
                pickle.dump([x, y, ch_names], file)
