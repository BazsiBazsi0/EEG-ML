# %%
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

print(__doc__)


# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1.0, 4.0
event_id = dict(hands=2, feet=3)
subjects = [1]
runs = [6, 10, 14]  # motor imagery: hands vs feet

raws = []
for subject in subjects:
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # Apply band-pass filter
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    raws.append(raw)

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
)
epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
labels = epochs.events[:, -1] - 2

# %%
# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Fit the CSP (Common Spatial Pattern) transformer and transform the training data
csp.fit(epochs_data_train, labels)
transformed_epochs = csp.transform(epochs_data_train)

# Visualize the data before CSP
plt.figure(figsize=(10, 5))
plt.title("Data Before CSP")
plt.plot(epochs_data_train[0])
plt.show()


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
plt.show()

# Fit the LDA (Linear Discriminant Analysis) classifier on the transformed data
lda.fit(transformed_epochs, labels)

# Transform the data using LDA
lda_transformed_data = lda.transform(transformed_epochs)

# Visualize the data after LDA
plt.figure(figsize=(10, 5))
plt.title("Data After LDA")
plt.scatter(lda_transformed_data, labels)
plt.xlabel("LDA Component")
plt.ylabel("Class Labels")
plt.show()

# %%

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(
    "Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance)
)
# %%

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

sfreq = raw.info["sfreq"]
w_length = int(sfreq * 0.5)  # running classifier: window length
w_step = int(sfreq * 0.1)  # running classifier: window step size
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

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
        X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
        score_this_window.append(lda.score(X_test, y_test))
    scores_windows.append(score_this_window)

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
    xytext=(max_score_time, max_score + 0.05),
    arrowprops=dict(facecolor="red", shrink=0.05),
)

plt.show()

# %%
