The dataset was created from the labels os PhysioNet Movement/Imagery dataset. The dataset was created by using the following steps:
- Extraction of labels from the dataset using MNE
- Filtering between 0 and 38 Hz
- Using the whole 5 second epochs
- Transitioning to numpy after MNE processing
- Creation of x and y datasets, where x is the EEG data samples and y is the labels
- Creation of a numpy array with the labels
- Saving the x and y datasets as numpy arrays

Additionally the dataset the number of classes either equalized or oversampled using SMOTE. The dataset was then split into training and testing sets. The training set was used to train the models, and the testing set was used to evaluate the models. The dataset was then saved as a numpy array for future use.