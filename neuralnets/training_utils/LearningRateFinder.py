import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class LearningRateFinder(tf.keras.callbacks.Callback):
    """
    Usage
        lr_finder = LearningRateFinder()
        model.fit(X_train, y_train, batch_size=512, callbacks=[lr_finder])
    """

    def __init__(self, start_lr=1e-10, end_lr=1e1, steps=1000):
        super(LearningRateFinder, self).__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.steps = steps
        self.lr_mult = (end_lr / start_lr) ** (1 / steps)
        self.history = {"lr": [], "loss": []}

    def on_train_begin(self, logs=None):
        self.best_loss = 1e9
        tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, batch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history["lr"].append(lr)

        loss = logs["loss"]
        self.history["loss"].append(loss)

        if batch > 5 and (np.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        lr *= self.lr_mult
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def plot_lr_finder(self, save_path="plot.png"):
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
