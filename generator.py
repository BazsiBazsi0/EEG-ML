import os
import numpy as np
import mne


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
        subjects = [str(s) for s in subjects]
        runs = [str(r) for r in runs]
        # selecting imaginary tasks, we only need these to train the network
        task2 = [4, 8, 12]  # fists
        task4 = [6, 10, 14]  # legs
        # iterating through the subjects list(a simple str list)
        for sub in subjects:
            # naming scheme for the folders
            if len(sub) == 1:
                sub_name = "S00" + sub
            elif len(sub) == 2:
                sub_name = "S0" + sub
            else:
                sub_name = "S" + sub
            sub_folder = os.path.join(data_path, sub_name)
            single_subject_run = []
            for run in runs:
                if len(run) == 1:
                    path_run = os.path.join(sub_folder, sub_name + "R0" + run + ".edf")
                else:
                    path_run = os.path.join(sub_folder, sub_name + "R" + run + ".edf")
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
                            raw_run.annotations.description[index] = "re"
                        if an == "T1":
                            raw_run.annotations.description[index] = "le"
                        if an == "T2":
                            raw_run.annotations.description[index] = "ri"
                if int(run) in task4:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "re"
                        if an == "T1":
                            raw_run.annotations.description[index] = "fi"
                        if an == "T2":
                            raw_run.annotations.description[index] = "fe"
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

        # TODO do we need channel selector?

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

    # TODO redundant lines
    @staticmethod
    def generate():
        exclude = [38, 88, 89, 92, 100, 104]
        # smartypants list comprehension for making a list with exclusion
        subjects = [n for n in np.arange(1, 50) if n not in exclude]
        # subjects = [1]
        runs = [4, 6, 8, 10, 12, 14]
        data_path = os.getcwd()
        save_path = os.path.join(os.getcwd(), "test_generate2/first_gen")
        # os.makedirs(save_path)
        for sub in subjects:
            x, y = Generator.load_data(subjects, runs, data_path)

            np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
            np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
