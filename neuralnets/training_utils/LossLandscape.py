import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.losses import MeanSquaredError  # type: ignore
from tensorflow.keras.models import Model  # type: ignore


class LossLandscape:
    """
    Class for computing and plotting the loss landscape of a neural network model.

    Args:
        model (Model): The neural network model.
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        h (float, optional): The step size for computing the loss landscape. Defaults to 0.01.
        num_points (int, optional): The number of points in each direction for computing the loss landscape. Defaults to 50.

    Attributes:
        model (Model): The neural network model.
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        h (float): The step size for computing the loss landscape.
        num_points (int): The number of points in each direction for computing the loss landscape.
        loss_fn (Loss): The loss function used for computing the loss.

    Methods:
        compute_loss(weights): Computes the loss for a given set of weights.
        plot_loss_landscape(): Plots the loss landscape.

    Example usage:
        ls = LossLandscape(model, x[:1], y[:1])
        ls.plot_loss_landscape()
    """

    def __init__(
        self,
        model: Model,
        x: np.ndarray,
        y: np.ndarray,
        h: float = 0.01,
        num_points: int = 50,
    ) -> None:
        self.model = model
        self.x = x.reshape(
            (
                int(x.shape[0] * x.shape[1]),
                x.shape[2],
                x.shape[3],
            )
        )
        self.y = y.reshape((int(y.shape[0] * y.shape[1]), y.shape[2]))
        self.h = h
        self.num_points = num_points
        self.loss_fn = MeanSquaredError()

    def compute_loss(self, weights: list[np.ndarray]) -> float:
        """
        Computes the loss for a given set of weights.

        Args:
            weights (list[np.ndarray]): The weights to compute the loss with.

        Returns:
            float: The computed loss.

        """
        original_weights = self.model.get_weights()
        self.model.set_weights(weights)
        y_pred = self.model.predict(self.x, verbose=0)
        loss = self.loss_fn(self.y, y_pred).numpy()
        self.model.set_weights(original_weights)
        return loss

    def plot_loss_landscape(self) -> None:
        """
        Plots the loss landscape.

        """
        original_weights = self.model.get_weights()
        weights_flat = np.concatenate([w.flatten() for w in original_weights])

        direction1 = np.random.normal(size=weights_flat.shape)
        direction1 /= np.linalg.norm(direction1)
        direction2 = np.random.normal(size=weights_flat.shape)
        direction2 -= direction2.dot(direction1) * direction1
        direction2 /= np.linalg.norm(direction2)

        loss_landscape = np.zeros((self.num_points, self.num_points))
        for i, alpha in enumerate(np.linspace(-1, 1, self.num_points)):
            for j, beta in enumerate(np.linspace(-1, 1, self.num_points)):
                weights = [
                    original_weights[k]
                    + alpha * self.h * direction1[k]
                    + beta * self.h * direction2[k]
                    for k in range(len(original_weights))
                ]
                loss_landscape[i, j] = self.compute_loss(weights)

        alpha_values, beta_values = np.meshgrid(
            np.linspace(-1, 1, self.num_points), np.linspace(-1, 1, self.num_points)
        )

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(alpha_values, beta_values, loss_landscape, cmap="viridis")
        ax.set_title("Loss Landscape")
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Loss")
        ax.view_init(elev=30, azim=45)
        plt.savefig("loss_landscape.png")
        plt.show()
