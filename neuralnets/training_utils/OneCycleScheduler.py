import tensorflow as tf
from typing import Optional


class OneCycleScheduler(tf.keras.callbacks.Callback):
    """
    A callback class for implementing the One Cycle Learning Rate Policy.

    Args:
        max_lr (float): The maximum learning rate.
        steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The total number of epochs.
        div_factor (float, optional): The factor by which the maximum learning rate is divided. Defaults to 25.0.
        pct_start (float, optional): The percentage of total steps to reach the maximum learning rate. Defaults to 0.3.
    Usage:
        scheduler = OneCycleScheduler(max_lr=0.01, steps_per_epoch=len(X_train) // batch_size, epochs=epochs)
        model.fit(X_train, y_train, epochs=epochs, callbacks=[scheduler])
        NOTE: document + div_factor was raised to 100.0 from 25.0
    """

    def __init__(
        self,
        max_lr: float,
        steps_per_epoch: int,
        epochs: int,
        div_factor: Optional[
            float
        ] = 100.0,  # lower learning rate start/end increase LR steepness
        pct_start: Optional[float] = 0.3,
        verbose: Optional[int] = 0,
        patience: int = 10,
        adaptive_stopping: bool = False,
    ):
        super(OneCycleScheduler, self).__init__()
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.total_steps = steps_per_epoch * epochs
        self.step = 0
        self.verbose = verbose
        self.patience = patience
        self.best_weights = None
        self.best_loss = float("inf")
        self.wait = 0
        self.last_round = 0
        self.adaptive_stopping = adaptive_stopping
        self.pct_start_steps = self.total_steps * pct_start
        self.pct_end_steps = self.total_steps * (1 - pct_start)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: OneCycleScheduler set learning rate to {lr}.")

        current_loss = logs.get("val_loss")
        if self.adaptive_stopping:
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
                        # Speed up learning rate decay by increasing div_factor
                        self.div_factor *= 10
                        self.wait = 0
                        self.last_round = 1
                        print(
                            f"Restored model weights from the end of the best epoch. Best loss: {self.best_loss:.4f}"
                        )

                    elif self.last_round == 1:
                        # Stop training and restore best weights
                        print(
                            f"Early stopping conditions reached, best weights restored. Best loss: {self.best_loss:.4f}"
                        )
                        self.model.stop_training = True
                        self.model.set_weights(self.best_weights)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Callback function called at the beginning of each training batch.

        Args:
            batch (int): The current batch index.
            logs (dict, optional): Dictionary containing the training metrics. Defaults to None.
        """
        if self.step <= self.pct_start_steps:
            scale = self.step / self.pct_start_steps
        else:
            scale = (self.total_steps - self.step) / self.pct_end_steps

        lr = self.max_lr * (1 + scale * (self.div_factor - 1)) / self.div_factor
        self.model.optimizer.lr = lr
        self.step += 1
