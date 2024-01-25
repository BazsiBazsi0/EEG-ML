import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K


class EarlyStoppingLearningRateSpeedup(callbacks.Callback):
    """Custom callback for early stopping with learning rate speedup.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
    """

    def __init__(self, patience: int = 20):
        super(EarlyStoppingLearningRateSpeedup, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.best_loss = float("inf")
        self.wait = 0
        self.last_round = 0

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Method called at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary containing the metrics results for this epoch.
        """
        current_loss = logs.get("val_loss")
        if current_loss < self.best_loss:
            # Update best loss and weights
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.last_round == 0:
                    # restore best weights, and speed up learning rate decay
                    self.model.set_weights(self.best_weights)
                    learning_rate = K.get_value(self.model.optimizer.lr)
                    K.set_value(self.model.optimizer.lr, learning_rate * 0.1)
                    print(
                        f"Learning rate changed from {learning_rate:.4f} to {learning_rate * 0.1:.4f}"
                    )
                    print("Restored model weights from the end of the best epoch.")
                    self.wait = 0
                    self.last_round = 1
                elif self.last_round == 1:
                    # Stop training and restore best weights
                    print(
                        "Early stopping reached in the second round of reduced lr training."
                    )
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)
                    print("Restored model weights from the end of the best epoch.")
