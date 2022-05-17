import main
import os
import sys
import numpy as np
import tensorflow as tf


class loader:

    # TODO redundant lines
    @staticmethod
    def load_saved_files():
        # load the saved files
        exclude = [38, 88, 89, 92, 100, 104]
        subjects = [n for n in np.arange(1, 50) if n not in exclude]
        xs = list()
        ys = list()
        data_x = list()
        data_y = list()
        for s in subjects:
            xs.append(np.load("test_generate2/first_gen/" + "x" + "_sub_" + str(s) + ".npy"))
            ys.append(np.load("test_generate2/first_gen/" + "y" + "_sub_" + str(s) + ".npy"))

        data_x.append(np.concatenate(xs))
        data_y.append(np.concatenate(ys))

        return np.concatenate(data_x), np.concatenate(data_y)