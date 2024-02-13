from matplotlib import pyplot as plt
import math
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
import numpy as np


class LRFinder:
    def __init__(self, model):
        """
        Initializes the LRFinder object.

        Parameters:
            model: The Keras model for which the learning rate finder will be used.
        """
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        """
        Callback function called at the end of each batch during model training.
        Logs the learning rate and loss, and checks for early stopping conditions.

        Parameters:
            batch: The batch index.
            logs: Dictionary containing the metrics for the current batch.
        """
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs["loss"]
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(
        self, x_train, y_train, start_lr, end_lr, batch_size=32, epochs=10, **kw_fit
    ):
        """
        Finds the optimal learning rate range for the model.

        Parameters:
            x_train: The input data for training.
            y_train: The target data for training.
            start_lr: The initial learning rate.
            end_lr: The final learning rate.
            batch_size: The batch size.
            epochs: The number of epochs.
            **kw_fit: Additional keyword arguments to be passed to the model's fit method.
        """
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (
            float(1) / float(num_batches)
        )

        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(
            on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs)
        )

        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[callback],
            **kw_fit
        )

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale="log"):
        """
        Plots the loss.

        Parameters:
            n_skip_beginning: Number of batches to skip on the left.
            n_skip_end: Number of batches to skip on the right.
            x_scale: Scale of the x-axis. Default is "log" scale.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(
            self.lrs[n_skip_beginning:-n_skip_end],
            self.losses[n_skip_beginning:-n_skip_end],
        )
        plt.xscale(x_scale)
        plt.show()

    def get_best_lr(self, n_skip_beginning=10, n_skip_end=5):
        """
        Returns the learning rate with the lowest loss.

        Parameters:
            n_skip_beginning: Number of batches to skip on the left.
            n_skip_end: Number of batches to skip on the right.

        Returns:
            The learning rate with the lowest loss.
        """
        best_loss_idx = np.argmin(self.losses[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_loss_idx]
