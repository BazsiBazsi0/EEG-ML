import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

from plotter import CSP_plotter

print(__doc__)


class CSP_Example:
    def __init__(self):
        # avoid classification of evoked responses by using epochs that start 1s after
        # cue onset. The orig was -1.0 for some reason
        self.tmin, self.tmax = 1, 4.0
        self.event_id = dict(hands=2, feet=3)
        self.subject = 2
        self.runs = [6, 10, 14]  # motor imagery: hands vs feet
        self.raws = []
        self.scores = []
        self.epochs_data = None

    def dataset_processor(self):
        raw_fnames = eegbci.load_data(self.subject, self.runs)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Apply band-pass filter
        raw.filter(7.0, 14, fir_design="firwin", skip_by_annotation="edge")

        self.raws.append(raw)

        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

        picks = pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )

        # Read epochs (train will be done only between 1 and 2s)
        # Testing will be done with a running classifier
        epochs = Epochs(
            raw,
            events,
            self.event_id,
            self.tmin,
            self.tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )
        # We crop the time between second 1 and 2, because thats the most probable time for the
        # event to happen and produce an event in the EEG signal
        epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
        # We are only interested in events 2 - hands and 3 - feet - we subtract 2 to have a binary
        # therefore this is a binary classification problem
        labels = epochs.events[:, -1] - 2

        return epochs, epochs_train, labels, raw

    def csp_processor(self, epochs, epochs_train, labels):
        # Define a monte-carlo cross-validation generator (reduce variance):
        # Monte-carlo cross-validation is just a fancy name for the process of
        # repeating the train-test split multiple times

        self.epochs_data = epochs.get_data()
        epochs_data_train = epochs_train.get_data()
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        cv_split = cv.split(epochs_data_train)

        # Assemble a classifier
        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        # Fit the CSP (Common Spatial Pattern) transformer and transform the training data
        csp.fit(epochs_data_train, labels)
        transformed_epochs = csp.transform(epochs_data_train)

        return transformed_epochs, cv, cv_split, csp, lda, epochs_data_train, labels

    def lda_processor(self, transformed_epochs, labels, lda):
        # Fit the LDA (Linear Discriminant Analysis) classifier on the transformed data
        lda.fit(transformed_epochs, labels)

        # Transform the data using LDA
        lda_transformed_data = lda.transform(transformed_epochs)

        return lda_transformed_data

    def pipeline(self, csp, lda, epochs_data_train, labels):
        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([("CSP", csp), ("LDA", lda)])
        scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

        # Printing the results
        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1.0 - class_balance)
        print(
            "\n \n Classification accuracy: %f / Chance level: %f"
            % (np.mean(scores), class_balance)
        )
        # csp.fit_transform(self.epochs_data, labels)

        return csp

    def fit(self, raw, cv_split, csp, lda, labels, epochs_data_train):
        sfreq = raw.info["sfreq"]
        w_length = int(sfreq * 0.5)  # running classifier: window length
        w_step = int(sfreq * 0.1)  # running classifier: window step size
        # 5 sec epoch 160 hz sampling rate - self.epochs_data.shape[2] = 800(+1)
        w_start = np.arange(0, self.epochs_data.shape[2] - w_length, w_step)

        scores_windows = []

        for train_idx, test_idx in cv_split:
            y_train, y_test = labels[train_idx], labels[test_idx]

            X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
            X_test = csp.transform(epochs_data_train[test_idx])

            # fit classifier
            lda.fit(X_train, y_train)

            # running classifier: test classifier on sliding window
            score_this_window = []
            for n in w_start:
                X_test = csp.transform(
                    self.epochs_data[test_idx][:, :, n : (n + w_length)]
                )
                score_this_window.append(lda.score(X_test, y_test))
            scores_windows.append(score_this_window)

        CSP_plotter.plot_scores(epochs, w_start, w_length, sfreq, scores_windows)


if __name__ == "__main__":
    csp_obj = CSP_Example()
    # Process the dataset and obtain epochs, epochs_train, labels, and raw data
    epochs, epochs_train, labels, raw = csp_obj.dataset_processor()
    # Process the epochs using CSP and obtain transformed_epochs, cv, cv_split, csp, lda, epochs_data_train, and labels
    transformed_epochs, cv, cv_split, csp, lda, epochs_data_train, labels = (
        csp_obj.csp_processor(epochs, epochs_train, labels)
    )
    # Process the transformed epochs using LDA and obtain lda_transformed_data
    lda_transformed_data = csp_obj.lda_processor(transformed_epochs, labels, lda)
    # Run the pipeline with CSP and LDA, and print the classification accuracy
    csp_obj.pipeline(csp, lda, epochs_data_train, labels)

    # Plot the CSP components
    CSP_plotter.plot_csp(csp, epochs, epochs.get_data(), labels)

    # Fit the CSP, LDA, and the classifier on the raw data
    # csp_obj.fit(raw, cv_split, csp, lda, labels, epochs_data_train)

    """# Plot the CSP patterns
    CSP_plotter.plot_csp_patterns(epochs_data_train)
    # Plot the CSP data after transformation
    CSP_plotter.plot_cst_data_after(transformed_epochs, labels)
    """
    # Plot the LDA data after transformation
    CSP_plotter.plot_lda_data_after(lda_transformed_data, labels)
