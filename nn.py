import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import visualkeras

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
    # More info/reading: https://imbalanced-learn.org/stable/over_sampling.html
    # Try ADASYN to see whats the result
    @staticmethod
    def smote_processor(x, y):
        sm = SMOTE(random_state=42)
        x_res, y_res = sm.fit_resample(x, y)
        return x_res, y_res

    @staticmethod
    def basic_net():
        inputs = tf.keras.Input(shape=(641, 64))
        spacing = visualkeras.SpacingDummyLayer(spacing=100)(inputs)
        conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding="same")(spacing)
        dense1 = tf.keras.layers.Dense(16, activation='relu')(conv1)
        mpool1d = tf.keras.layers.MaxPooling1D()(dense1)
        flat = tf.keras.layers.Flatten()(mpool1d)
        out = tf.keras.layers.Dense(3, activation='softmax')(flat)

        return tf.keras.Model(inputs, out)

    @staticmethod
    def starter_net():
        kernel_size_0 = 20
        kernel_size_1 = 6
        drop_rate = 0.5
        # 20512 is the input if i include all the ch-s
        # M x N M: length of the time window(160 x 4)
        #       N: number of EEG ch-s
        # with only one pair of electrodes it requires 641x2, with all the electrodes: 641x64
        inputs = tf.keras.Input(shape=(641, 2))
        spacing = visualkeras.SpacingDummyLayer(spacing=100)(inputs)

        conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="same")(spacing)
        batch_n_1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="valid")(batch_n_1)
        batch_n_2 = tf.keras.layers.BatchNormalization()(conv2)
        spatial_drop1 = tf.keras.layers.SpatialDropout1D(drop_rate)(batch_n_2)
        conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(spatial_drop1)
        spacing2 = visualkeras.SpacingDummyLayer(spacing=100)(conv3)

        avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)(spacing2)
        conv4 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(avg_pool1)
        spatial_drop_2 = tf.keras.layers.SpatialDropout1D(drop_rate)(conv4)
        spacing3 = visualkeras.SpacingDummyLayer(spacing=100)(spatial_drop_2)


        flat = tf.keras.layers.Flatten()(spacing3)
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
        inputs = tf.keras.Input(shape=(64, 641))
        tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        return 0

    # TODO fix
    # Something fishy is going on, shows error
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