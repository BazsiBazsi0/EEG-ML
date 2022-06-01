import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import loader
import nn
import generator

#generator.Generator.generate()

x, y = loader.FileLoader.load_saved_files()
print(np.shape(x), np.shape(y))  # 50 subjects: x = (8640, 2, 641) y = (8640,) if all electrodes: (1620, 64, 641) (1620,)

# TODO mne plotting for sanity check plot one or two

# Transform y to one-hot-encoding
y_one_hot = nn.NeuralNets.to_one_hot(y)
# Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
# print(np.shape(reshaped_x))  # (8640, 1282) this makes this into an "1D" array

# Does train/test split with scikit learn, test size 0.2 stratification done on y
x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x, y_one_hot,
                                                                                stratify=y_one_hot, test_size=0.20,
                                                                                random_state=42)
# New loading
#x_train_raw, y_train_raw, x_test_raw, y_test_raw = loader.FileLoader.load_saved_files_new()

# Scale train/test
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)
# print(x_test_valid_scaled_raw)

# Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                            y_valid_test_raw,
                                                            stratify=y_valid_test_raw,
                                                            test_size=0.50,
                                                            random_state=42)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1] / 2), 2)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1] / 2), 2)

# SMOTE, Synthetic Minority Oversampling Technique
# TODO classes seem to be the same before and after smote
x_train_smote_raw, y_train = nn.NeuralNets.smote_processor(x_train_scaled_raw, y_train_raw)
print('classes count')
print('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1] / 2), 2)

print('Final shape of x_train: ', np.shape(x_train)) #shape: (6912, 641, 2)
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()

# TODO visualize and understand this, try at least 2 other structures and generate results
model = nn.NeuralNets.starter_net()
with open('modelsummary.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'),line_length=120)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# TODO save txt, model summary, other descriptors
# Confusion matrix is built in by sklearn, implemented in as metrics_generation in nn
model.fit(x_train, y_train, epochs=1)

history = model.fit(x_train, y_train, epochs=1, batch_size=10, validation_data=(x_valid, y_valid))

testLoss, testAcc = model.evaluate(x_test, y_test)
print('Predicted test accuracy: ', testAcc)
print('Predicted test loss: ', testLoss)

# TODO train set equal amount of examples in classes
nn.NeuralNets.metrics_generation(model, x_test, y_test)

