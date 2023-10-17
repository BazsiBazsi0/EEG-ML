import numpy as np

class FileLoader:
    def __init__(self, patients=10, subjects=range(60, 70)):
        self.patients = patients
        self.subjects = subjects

    def load_saved_files_val(self):
        xs, ys = [], []
        for s in self.subjects:
            # level 0 7, level 1
            xs.append(
                np.load("dataset/filtered_data/ch_level_1/x_sub_{}.npy".format(s))
            )
            ys.append(
                np.load("dataset/filtered_data/ch_level_1/y_sub_{}.npy".format(s))
            )

        x_concat = np.concatenate(xs)
        y_concat = np.concatenate(ys)

        # Pre-processing the variables
        x_pre, y_pre, x_no_smote_pre, y_no_smote_pre = self.preprocessor(
            x_concat, y_concat
        )

        num_patients = x_pre.shape[0] // self.patients
        x = x_pre.reshape(
            self.patients, num_patients, x_pre.shape[1], x_pre.shape[2]
        ).astype(np.float32)
        y = y_pre.reshape(self.patients, num_patients, y_pre.shape[1]).astype(
            np.float32
        )

        x_no_smote = x_no_smote_pre.reshape(
            self.patients,
            x_no_smote_pre.shape[0] // self.patients,
            x_no_smote_pre.shape[1],
            x_no_smote_pre.shape[2],
        ).astype(np.float32)

        y_no_smote = y_no_smote_pre.reshape(
            self.patients,
            y_no_smote_pre.shape[0] // self.patients,
            y_no_smote_pre.shape[1],
        ).astype(np.float32)

        return x, y, x_no_smote, y_no_smote
