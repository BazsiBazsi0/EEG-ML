import os
from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


class SubjectsClass:
    def __init__(self, debug, subject_s={}, subject_names=[], recordings={}, subj_single=''):
        self.debug = None

    @staticmethod
    def list_files(self):
        # This only lists files in cwd
        # prints the files found
        # maybe use for sanity check later
        path_of_the_directory = os.getcwd()
        # path_of_the_directory= 'EEG_Physionet'
        print("Files and directories in a specified path:")
        for filename in os.listdir(path_of_the_directory):
            f = os.path.join(path_of_the_directory, filename)
            if os.path.isfile(f):
                print(f)

    @staticmethod
    def read_dir_files(self):
        # this reads in folders and files
        # python 3.5 style with scandir
        # the files & dirs in the root(current working dir) are saved in arrays
        # prints the dirs found, but the object are not simple strings
        # I can't make use of it currently

        folders = []
        folder_paths = []
        files = []
        for entry in os.scandir(os.getcwd()):
            if entry.is_dir():
                folders.append(entry)
                folder_paths.append(entry.path)
            elif entry.is_file():
                files.append(entry.path)
        print('Folders:')
        for f in folders:
            print(f)

    def dir_list(self):
        # this takes the list of dirs and put them into a list
        dirlist = [item for item in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), item))]

        if self.debug:
            print(dirlist)

        return dirlist

    def open_subj_files(self, subj_single):
        # This lists the files in certain subject dir, filters *.edf files for safety
        source_dir = Path(subj_single)
        filelist = os.listdir(source_dir)

        for f in filelist:
            raw = mne.io.read_raw_edf(f)
            events, event_dict = mne.events_from_annotations(raw)
            event_dict = dict(rest=1, left=2, right=3)
            epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=0, tmax=4, baseline=None)

        # below is file processing if needed in any case
        """
        files = source_dir.glob('*.edf')
        for file in files:
            with file.open('r') as file_handle:
                for single_file in file_handle:
                    # do your thing
                    yield single_file
        """

        if self.debug:
            print(filelist)

        return 0

    # TODO redundant lines
    @staticmethod
    def loader():
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

    # stolen-onehot to test the process
    # TODO rewrite this block+investigate
    @staticmethod
    def to_one_hot(y, by_sub=False):
        if by_sub:
            new_array = np.array(["nan" for nan in range(len(y))])
            for index, label in enumerate(y):
                new_array[index] = ''.join([i for i in label if not i.isdigit()])
        else:
            new_array = y.copy()
        total_labels = np.unique(new_array)
        mapping = {}
        for x in range(len(total_labels)):
            mapping[total_labels[x]] = x
        for x in range(len(new_array)):
            new_array[x] = mapping[new_array[x]]

        return tf.keras.utils.to_categorical(new_array)


x, y = SubjectsClass.loader()
print(np.shape(x), np.shape(y))  # x = (8640, 2, 641) y = (8640,)
# TODO ask AA about how to check my data at this point, maybe scatterplot
# TODO onehot
# TODO reshape/scale
# Transform y to one-hot-encoding
y_one_hot = SubjectsClass.to_one_hot(y, by_sub=False)
# Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(np.shape(reshaped_x))

x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x, y_one_hot,
                                                                                stratify=y_one_hot, test_size=0.20,
                                                                                random_state=42)

# Scale independently train/test
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)
print(x_test_valid_scaled_raw)

# Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                            y_valid_test_raw,
                                                            stratify=y_valid_test_raw,
                                                            test_size=0.50,
                                                            random_state=42)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1] / 2), 2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1] / 2), 2).astype(np.float64)

# apply smote to train data
print('classes count')
print('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
# smote
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
print('classes count')
print('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1] / 2), 2).astype(
    np.float64)

loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()

kernel_size_0 = 20
kernel_size_1 = 6
drop_rate = 0.5

# TODO try something else
inputs = tf.keras.Input(shape=(641, 2))
conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="same")(inputs)
batch_n_1 = tf.keras.layers.BatchNormalization()(conv1)
conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="valid")(batch_n_1)
batch_n_2 = tf.keras.layers.BatchNormalization()(conv2)
spatial_drop1 = tf.keras.layers.SpatialDropout1D(drop_rate)(batch_n_2)
conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(spatial_drop1)
avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)(conv3)
conv4 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(avg_pool1)
spatial_drop_2 = tf.keras.layers.SpatialDropout1D(drop_rate)(conv4)
flat = tf.keras.layers.Flatten()(spatial_drop_2)
dense1 = tf.keras.layers.Dense(296, activation='relu')(flat)
dropout1 = tf.keras.layers.Dropout(drop_rate)(dense1)
dense2 = tf.keras.layers.Dense(148, activation='relu')(dropout1)
dropout2 = tf.keras.layers.Dropout(drop_rate)(dense2)
dense3 = tf.keras.layers.Dense(74, activation='relu')(dropout2)
dropout3 = tf.keras.layers.Dropout(drop_rate)(dense3)
out = tf.keras.layers.Dense(5, activation='softmax')(dropout3)
model = tf.keras.Model(inputs, out)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# subj = SubjectsClass(1)
# subjects.subject_names = subjects.dir_list()
# subj.dir_list()
# subj.open_subj_files('S001')
