import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

class NeuralNets:

    # if I want to switch this to another method here are the possibilities:
    # use sklearn onehot, this could cause problems down the line, need to investigate
    # use tf.one_hot but this doesn't preserve the labels, since it's a integer/tensor operation
    # use this with minor improvements as this already uses the .to_categorical method
    @staticmethod
    def to_one_hot(y):
        # shallow copy to a new array
        y_shallow_copy = y.copy()
        # New unique labels in case of double vals(maybe there are duplicates)
        total_labels = np.unique(y_shallow_copy)

        # Dictionary named encoding for labels
        encoding = {}
        for x in range(len(total_labels)):
            encoding[total_labels[x]] = x
        for x in range(len(y_shallow_copy)):
            y_shallow_copy[x] = encoding[y_shallow_copy[x]]

        return tf.keras.utils.to_categorical(y_shallow_copy)

    # Synthetic Minority Oversampling Technique
    # This balances the imbalance between 'rest' (re) and the other classes
    # TODO more reading: https://imbalanced-learn.org/stable/over_sampling.html
    @staticmethod
    def smote_processor(x, y):
        sm = SMOTE(random_state=42)
        sm.fit_resample(x, y)
        return x,y

    @staticmethod
    def starter_net():
        kernel_size_0 = 20
        kernel_size_1 = 6
        drop_rate = 0.5
        # it wants to have 20512 as input if i include all the ch-s, makes no sense it should have 641 x 64 or similar
        # M x N M: length of the time window(160 x 4)
        #       N: number of EEG ch-s
        # with only one pair of electrodes it requires 641x2
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

    @staticmethod
    def simplifed_starter_net():
        kernel_size_0 = 20
        kernel_size_1 = 6
        drop_rate = 0.5

        inputs = tf.keras.Input(shape=(641, 2))
        conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="same")(inputs)
        batch_n_1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="valid")(
            batch_n_1)
        batch_n_2 = tf.keras.layers.BatchNormalization()(conv2)
        #spatial_drop1 = tf.keras.layers.SpatialDropout1D(drop_rate)(batch_n_2)
        conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(
            batch_n_2)
        avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)(conv3)
        conv4 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(
            avg_pool1)
        #spatial_drop_2 = tf.keras.layers.SpatialDropout1D(drop_rate)(conv4)
        flat = tf.keras.layers.Flatten()(conv4)
        dense1 = tf.keras.layers.Dense(296, activation='relu')(flat)
        #dropout1 = tf.keras.layers.Dropout(drop_rate)(dense1)
        dense2 = tf.keras.layers.Dense(148, activation='relu')(dense1)
        #dropout2 = tf.keras.layers.Dropout(drop_rate)(dense2)
        dense3 = tf.keras.layers.Dense(74, activation='relu')(dense2)
        #dropout3 = tf.keras.layers.Dropout(drop_rate)(dense3)
        out = tf.keras.layers.Dense(5, activation='softmax')(dense3)

        return tf.keras.Model(inputs, out)

    # Next evolution step after AlexNet, is addresses the depth issue and implements various other improvements.
    # Created by the Visual Geometry Group (University of Oxford)
    # 92.7% top-5 test accuracy in ImageNet
    # TODO needs 2D input
    @staticmethod
    def VGG_net_16():
        inputs = tf.keras.Input(shape=(641, 2))
        conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")
        return 0

    # TODO Extend this s little bit add more metrics
    # Something fishy is going on, reimplement Ch-s fully and check again
    @staticmethod
    def metrics_generation(model, x_test, y_test):
        y_predict = model.predict(x_test)

        # convert from one hot encode in string
        y_test_class = np.argmax(y_test, axis=1)
        y_predicted_class = np.argmax(y_predict, axis=1)

        print('Classification report: ')
        print(classification_report(y_test_class, y_predicted_class))
        #                            target_names=["rest", "left", "right", "fists", "feet"]))

        print('Confusion matrix: ')

        print(confusion_matrix(y_test_class, y_predicted_class))