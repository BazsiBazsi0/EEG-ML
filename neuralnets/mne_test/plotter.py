import numpy as np
import matplotlib.pyplot as plt


class CSP_plotter:

    @staticmethod
    def plot_scores(epochs, w_start, w_length, sfreq, scores_windows):
        # Plot scores over time
        w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

        plt.figure()
        plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
        plt.axvline(0, linestyle="--", color="k", label="Onset")
        plt.axhline(0.5, linestyle="-", color="k", label="Chance")
        plt.xlabel("time (s)")
        plt.ylabel("classification accuracy")
        plt.title("Classification score over time")
        plt.legend(loc="lower right")

        # Calculate the maximum score and its corresponding time
        max_score = np.max(np.mean(scores_windows, 0))
        max_score_time = w_times[np.argmax(np.mean(scores_windows, 0))]

        # Add annotation for the maximum score
        plt.annotate(
            "Max: {:.2f}".format(max_score),
            xy=(max_score_time, max_score),
            # Position of the text relative to the point positive puts the text above the point
            xytext=(max_score_time, max_score + 0.03),
            arrowprops=dict(facecolor="red", shrink=0.5, width=1, headwidth=5),
        )

        plt.show()

    @staticmethod
    def plot_csp_patterns(epochs_data_train):
        # Visualize the data before CSP
        plt.figure(figsize=(10, 5))
        plt.title("Data Before CSP")
        plt.plot(epochs_data_train[0])
        plt.show()

    @staticmethod
    def plot_csp(csp, epochs, epochs_data, labels):
        csp.fit_transform(epochs_data, labels)
        # plot CSP patterns estimated on full data for visualization
        csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

    @staticmethod
    def plot_cst_data_after(transformed_epochs, labels):
        # Calculate the average of each component for each class
        avg_class_0 = np.mean(transformed_epochs[labels == 0], axis=0)
        avg_class_1 = np.mean(transformed_epochs[labels == 1], axis=0)

        # Plot the averages
        plt.figure(figsize=(10, 5))
        plt.title("Average Component Values After CSP")
        plt.plot(avg_class_0, label="Class 0", marker="o")
        plt.plot(avg_class_1, label="Class 1", marker="o")
        plt.xlabel("Component")
        plt.ylabel("Average Value")
        plt.legend()
        plt.xticks(range(4))
        plt.xlim(0, 3)
        plt.show()

    @staticmethod
    def plot_lda_data_after(lda_transformed_data, labels):
        """
        This function creates a scatter plot of the LDA-transformed data. Each class is represented by a different color.
        If the LDA-transformed data has more than one component (i.e., it's a multi-class problem),
          the function plots the first component against the second component.
        If the LDA-transformed data has only one component (i.e., it's a binary problem),
          the function plots the component values against the sample indices.

        Args:
            lda_transformed_data (numpy.ndarray): The data after being transformed by LDA. Each row is a data point and each column is an LDA component.
            labels (numpy.ndarray): The class labels for the data points. The labels should be integers starting from 0.
        """
        # Visualize the data after LDA
        plt.figure(figsize=(10, 10))
        plt.title("Data After LDA")

        # Get unique labels and corresponding colors
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(unique_labels)))

        # Check the number of components in lda_transformed_data
        if lda_transformed_data.shape[1] > 1:
            # Multi-class problem: plot both components
            for label, color in zip(unique_labels, colors):
                plt.scatter(
                    lda_transformed_data[labels == label, 0],
                    lda_transformed_data[labels == label, 1],
                    c=color,
                    label=label,
                )
            plt.xlabel("First LDA Component")
            plt.ylabel("Second LDA Component")
        else:
            # Binary problem: plot only one component
            for label, color in zip(unique_labels, colors):
                plt.scatter(
                    np.arange(len(lda_transformed_data[labels == label])),
                    lda_transformed_data[labels == label, 0],
                    c=color,
                    label=label,
                )
            plt.xlabel("Samples")
            plt.ylabel("LDA Component")

        plt.legend()
        plt.show()
