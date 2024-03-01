from typing import List
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
from scipy import signal


class DataVisualizer:
    """
    A class used to visualize data in 3D using PCA.

    ...

    Attributes
    ----------
    X : np.ndarray
        The input data to be visualized
    y : np.ndarray
        The labels corresponding to the input data
    colors : List[str]
        The colors to be used for different classes in the visualization
    X_pca : np.ndarray
        The input data transformed using PCA

    Methods
    -------
    apply_pca():
        Applies PCA to the input data
    plot_3d():
        Plots the PCA transformed data in 3D
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Constructs all the necessary attributes for the DataVisualizer object.

        Parameters
        ----------
            X : np.ndarray
                The input data to be visualized
            y : np.ndarray
                The labels corresponding to the input data
        """
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.colors: List[str] = ["b", "g", "r", "c", "m"]
        self.X_pca: np.ndarray

    def apply_pca(self) -> None:
        """
        Applies PCA to the input data.

        The PCA is applied with 3 components and the result is stored in the X_pca attribute.
        """
        pca = PCA(n_components=3)
        self.X_pca = pca.fit_transform(self.X)

    def plot_3d(self) -> None:
        """
        Plots the PCA transformed data in 3D.

        The data is plotted in 3D with different colors for different classes.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Get the unique class labels
        classes = np.unique(self.y)

        # Ensure there are enough colors for each class
        if len(self.colors) < len(classes):
            self.colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

        # Plot each class with a different color
        for i, class_ in enumerate(classes):
            ax.scatter(
                self.X_pca[self.y == class_, 0],
                self.X_pca[self.y == class_, 1],
                self.X_pca[self.y == class_, 2],
                c=self.colors[i],
            )

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        plt.show()

    def scatter_plot(self) -> None:
        """
        Creates a scatter plot of the input data and labels.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.X, y=self.y, hue=self.y)
        plt.show()

    def histogram(self) -> None:
        """
        Creates a histogram of the input data and labels.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(x=self.X, hue=self.y, multiple="stack")
        plt.show()

    def pair_plot(self) -> None:
        """
        Creates a pair plot of the input data and labels.
        """
        sns.pairplot(self.X, hue=self.y)
        plt.show()

    def spectral_analysis(self, fs=160):
        """
        Perform spectral analysis on EEG data.

        Parameters:
        eeg_data : np.ndarray
            The EEG data to be analyzed. The shape is assumed to be (samples, channels, datapoints).
        fs : float
            The sampling frequency of the EEG data. Default is 160 Hz.
        """
        # Get the unique class labels
        classes = np.argmax(self.y, axis=1)
        unique_classes = np.unique(classes)

        # Compute the frequency and power spectral density (PSD) for each class
        plt.figure(figsize=(10, 6))
        for class_ in unique_classes:
            class_data = self.X[classes == class_]
            freqs, psd = signal.welch(np.mean(class_data, axis=0), fs, nperseg=1024)

            # Plot the average PSD for this class
            avg_psd = np.mean(psd, axis=0)
            plt.plot(freqs, avg_psd, label=f"Class {class_}")

        plt.title("Average Power Spectral Density of EEG Data by Class")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V^2/Hz)")
        plt.legend()
        plt.show()
