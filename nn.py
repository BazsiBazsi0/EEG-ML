import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE


class NeuralNets:

    # stolen-onehot to test the process
    # TODO rewrite this block + further investigate, use built in onehot
    @staticmethod
    def to_one_hot(y, by_sub=False):
        if by_sub:
            new_array = np.array(["nan" for nan in range(len(y))])
            for index, label in enumerate(y):
                new_array[index] = ''.join([i for i in label if not i.isdigit()])
        else:
            # hard copy to a new array
            new_array = y.copy()

        # New unique labels in case of double vals(maybe there are duplicates)
        total_labels = np.unique(new_array)

        # Mapping ofr labels(?)
        mapping = {}
        for x in range(len(total_labels)):
            mapping[total_labels[x]] = x
        for x in range(len(new_array)):
            new_array[x] = mapping[new_array[x]]

        return tf.keras.utils.to_categorical(new_array)

    @staticmethod
    def smote_processor(x, y):
        sm = SMOTE(random_state=42)
        sm.fit_resample(x, y)

    @staticmethod
    def starter_net():
        kernel_size_0 = 20
        kernel_size_1 = 6
        drop_rate = 0.5
        # TODO try something else
        inputs = tf.keras.Input(shape=(641, 2))
        conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="same")(inputs)
        batch_n_1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="valid")(
            batch_n_1)
        batch_n_2 = tf.keras.layers.BatchNormalization()(conv2)
        spatial_drop1 = tf.keras.layers.SpatialDropout1D(drop_rate)(batch_n_2)
        conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(
            spatial_drop1)
        avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)(conv3)
        conv4 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(
            avg_pool1)
        spatial_drop_2 = tf.keras.layers.SpatialDropout1D(drop_rate)(conv4)
        flat = tf.keras.layers.Flatten()(spatial_drop_2)
        dense1 = tf.keras.layers.Dense(296, activation='relu')(flat)
        dropout1 = tf.keras.layers.Dropout(drop_rate)(dense1)
        dense2 = tf.keras.layers.Dense(148, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(drop_rate)(dense2)
        dense3 = tf.keras.layers.Dense(74, activation='relu')(dropout2)
        dropout3 = tf.keras.layers.Dropout(drop_rate)(dense3)
        out = tf.keras.layers.Dense(5, activation='softmax')(dropout3)

        return tf.keras.Model(inputs, out)