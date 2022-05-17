import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import loader


class SubjectsClass:

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


x, y = loader.load_saved_files()
print(np.shape(x), np.shape(y))  # x = (8640, 2, 641) y = (8640,)
# TODO ask AA about how to check my data at this point, maybe scatter plot
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
