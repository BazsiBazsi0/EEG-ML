import numpy as np


class FileLoader:

    @staticmethod
    def load_saved_files():
        # load the saved files
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 20) if n not in exclude]
        xs = list()
        ys = list()
        data_x = list()
        data_y = list()
        for s in subjects:
            xs.append(np.load("legacy_generator/all_electrodes_50_patients/" + "x" + "_sub_" + str(s) + ".npy"))
            ys.append(np.load("legacy_generator/all_electrodes_50_patients/" + "y" + "_sub_" + str(s) + ".npy"))

        data_x.append(np.concatenate(xs))
        data_y.append(np.concatenate(ys))

        return np.concatenate(data_x), np.concatenate(data_y)