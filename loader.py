import numpy as np
import pickle
import numpy.random
import os


class FileLoader:

    @staticmethod
    def load_saved_files_new():
        """
        Before running this generate the data with generator_fix.py
        """

        path = "dataset/sub_by_sub_motor_imagery"

        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [str(n) for n in np.arange(1, 110) if n not in exclude]

        all_subs = list()

        for subject_number in subjects:
            with open(os.path.join(path, subject_number + ".pkl"), "rb") as file:
                all_subs.append(pickle.load(file))
                # each file is composed as follows:
                # [numpy array [n_task, channels, time], list of labels, ch_names]

        channels_mapping = {name: index for name, index in
                            zip(all_subs[0][2], range(len(all_subs[0][2])))}

        couples = [["FC1", "FC2"],
                   ["FC3", "FC4"],
                   ["FC5", "FC6"],
                   ["C5", "C6"],
                   ["C3", "C4"],
                   ["C1", "C2"],
                   ["CP1", "CP2"],
                   ["CP3", "CP4"],
                   ["CP5", "CP6"]]

        couples_mapping = [[channels_mapping[couple[0]],
                            channels_mapping[couple[1]]] for couple in couples]

        one_hot_encoding = {"R": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                            "L": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
                            "LR": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
                            "F": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
                            "B": np.array([0.0, 0.0, 0.0, 0.0, 1.0])}

        random = numpy.random.default_rng(42)

        x_train_raw, y_train_raw, x_test_raw, y_test_raw = list(), list(), list(), list()
        for sub in all_subs:
            for trial in range(len(sub[0])):
                if random.random() > 0.3:
                    for couple in couples_mapping:
                        x_train_raw.append(np.array([sub[0][trial, couple[0], :],
                                                     sub[0][trial, couple[1], :]]))

                        y_train_raw.append(one_hot_encoding[sub[1][trial]])
                else:
                    for couple in couples_mapping:
                        x_test_raw.append(np.array([sub[0][trial, couple[0], :],
                                                    sub[0][trial, couple[1], :]]))

                        y_test_raw.append(one_hot_encoding[sub[1][trial]])
        return x_train_raw, y_train_raw, x_test_raw, y_test_raw

    @staticmethod
    def load_saved_files():
        # load the saved files
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 10) if n not in exclude]
        xs = list()
        ys = list()
        data_x = list()
        data_y = list()
        for s in subjects:
            xs.append(np.load("test_generate/second_gen/" + "x" + "_sub_" + str(s) + ".npy"))
            ys.append(np.load("test_generate/second_gen/" + "y" + "_sub_" + str(s) + ".npy"))

        data_x.append(np.concatenate(xs))
        data_y.append(np.concatenate(ys))

        return np.concatenate(data_x), np.concatenate(data_y)
