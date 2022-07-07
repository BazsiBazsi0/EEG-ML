import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class NeuralNets:

    @staticmethod
    def leave_one_out(x, y):
        models = {'StarterNet': NeuralNets.starter_net(),
                  'BasicNet': NeuralNets.basic_net()}
        history_loo = []
        acc_loop = []
        acc = []
        avg_acc = []
        for model_name, model in models.items():
            for i in range(0, len(x)):
                print('Current loop index: ', i)
                # Leaving out one person from the training
                x_loop = np.append(x[:i], x[i + 1:], 0)
                # Reshaping into 2D array
                x_2d = x_loop.reshape((int(x_loop.shape[0] * x_loop.shape[1]), x_loop.shape[2], x_loop.shape[3]))

                y_loop = np.append(y[:i], y[i + 1:], 0)
                y_2d = y_loop.reshape((int(y_loop.shape[0] * y_loop.shape[1]), y_loop.shape[2]))

                # Validation data: the left out person from train data
                x_val = x[i]
                y_val = y[i]

                # Creating and saving the models

                # Validation is done on person left out on from the training from the train set
                history = model.fit(x_2d, y_2d, epochs=10, batch_size=100, validation_data=(x_val, y_val))
                history_loo.append(history)

                # Saving and averaging the accuracies, evaluation is done on the original dataset
                test_loss, test_acc = models[model_name].evaluate(x_val, y_val)

                acc_loop.append(test_acc)
                avg_acc_loop = [np.average(acc_loop)]

            acc.append(acc_loop)
            avg_acc.append(avg_acc_loop)
            acc_loop = []

        return models, history_loo, acc, avg_acc

    @staticmethod
    def starter_net():
        loss = tf.keras.losses.categorical_crossentropy
        optimizer = tf.keras.optimizers.Adam()

        kernel_size_0 = 20
        kernel_size_1 = 6
        drop_rate = 0.5

        inputs = tf.keras.Input(shape=(64, 641))
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

        model = tf.keras.Model(inputs, out)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def basic_net():
        loss = tf.keras.losses.categorical_crossentropy
        optimizer = tf.keras.optimizers.Adam()

        kernel_size_0 = 20
        kernel_size_1 = 6
        drop_rate = 0.5

        inputs = tf.keras.Input(shape=(64, 641))
        conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="same")(inputs)
        batch_n_1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_0, activation='relu', padding="valid")(
            batch_n_1)
        batch_n_2 = tf.keras.layers.BatchNormalization()(conv2)
        # spatial_drop1 = tf.keras.layers.SpatialDropout1D(drop_rate)(batch_n_2)
        conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(
            batch_n_2)
        avg_pool1 = tf.keras.layers.AvgPool1D(pool_size=2)(conv3)
        conv4 = tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size_1, activation='relu', padding="valid")(
            avg_pool1)
        # spatial_drop_2 = tf.keras.layers.SpatialDropout1D(drop_rate)(conv4)
        flat = tf.keras.layers.Flatten()(conv4)
        dense1 = tf.keras.layers.Dense(296, activation='relu')(flat)
        # dropout1 = tf.keras.layers.Dropout(drop_rate)(dense1)
        dense2 = tf.keras.layers.Dense(148, activation='relu')(dense1)
        # dropout2 = tf.keras.layers.Dropout(drop_rate)(dense2)
        dense3 = tf.keras.layers.Dense(74, activation='relu')(dense2)
        # dropout3 = tf.keras.layers.Dropout(drop_rate)(dense3)
        out = tf.keras.layers.Dense(5, activation='softmax')(dense3)

        model = tf.keras.Model(inputs, out)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def metrics_generation(model, x_test, y_test):
        y_predict = model.predict(x_test)

        # convert from one hot encode in string
        y_test_class = tf.argmax(y_test, axis=1)
        y_predicted_classes = tf.argmax(y_predict, axis=1)

        print('Classification report: ')
        print(classification_report(y_test_class, y_predicted_classes))
        #                            target_names=["rest", "left", "right", "fists", "feet"]))

        print('Confusion matrix: ')

        print(confusion_matrix(y_test_class, y_predicted_classes))

    @staticmethod
    def metrics_csv(file_name, acc, average_accuracy):
        np.savetxt(file_name, np.column_stack([acc[0], acc[1]]), delimiter=",", fmt='%.4f',
                   header='no_smote,smote', comments='')
        with open(file_name, 'a') as file:
            file.write('\n' + str(np.round(average_accuracy[0], 4)) + ',' + str(np.round(average_accuracy[1], 4)))

    @staticmethod
    def save_accuracy_curves(history, number_of_epochs, filename):
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        epochs_range = range(number_of_epochs)

        fig = plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="train accuracy")
        plt.plot(epochs_range, val_acc, label="validation accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="train loss")
        plt.plot(epochs_range, val_loss, label="validation loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        fig.tight_layout()
        plt.savefig(filename)
