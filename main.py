import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import loader
import nn
import visualkeras
import generator

# generator.Generator.generate()

x, y = loader.FileLoader.load_saved_files()
# print(np.shape(x), np.shape(y))

# Transform y to one-hot-encoding
y_one_hot = nn.NeuralNets.to_one_hot(y)
# Create one-dimensional vectors from each row of x input data
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
# print(np.shape(reshaped_x))

# Does train/test split with scikit learn, test size 0.2 stratification done on y
x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x, y_one_hot,
                                                                                stratify=y_one_hot, test_size=0.20,
                                                                               random_state=42)
# New generator
#generator.Generator.generate()

# New loading
x_train_raw, y_train_raw, x_test_raw, y_test_raw = loader.FileLoader.load_saved_files_new(max_subjects=10)
# print('Shape of x_train_raw after loading:' + np.shape(x_train_raw))
# Reshape for scaling
#reshaped_x_train = x_train_raw.reshape(x_train_raw.shape[0], x_train_raw.shape[1] * x_train_raw.shape[2])
#reshaped_x_test = x_test_raw.reshape(x_test_raw.shape[0], x_test_raw.shape[1] * x_test_raw.shape[2])
#print('Shape of x_train_raw after reshaping indo 1D array:' + str(np.shape(reshaped_x_train)))

# Scale train/test
#x_train_scaled_raw = minmax_scale(reshaped_x_train, axis=1)
#x_test_scaled_raw = minmax_scale(reshaped_x_test, axis=1)
# print(x_test_valid_scaled_raw)

# Create Validation/test
"""x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_scaled_raw,
                                                            y_test_raw,
                                                            stratify=y_test_raw,
                                                            test_size=0.50,
                                                            random_state=42)"""

# x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1] / 2), 2)
# x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1] / 2), 2)

# SMOTE, Synthetic Minority Oversampling Technique, requires 2D input
"""x_train_smote_raw, y_train = nn.NeuralNets.smote_processor(x_train_raw, y_train_raw)
print('Before SMOTE: ' + str(y_train_raw.sum(axis=0)) + '\n After SMOTE: ' + str(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1] / 2), 2)
print('Final shape of x_train: ', np.shape(x_train))  # shape: ('total lenght', 641, 2)"""

loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()

# TODO implement more nets to test
model = nn.NeuralNets.basic_net()
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'), line_length=120)

from PIL import ImageFont

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is definitely an option!
visualkeras.layered_view(model, legend=True, font=font, to_file='output.png')  # font is optional!"""

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# Confusion matrix is built in by sklearn, implemented in as metrics_generation in nn
model.fit(x_train_raw, y_train_raw, epochs=1)

history = model.fit(x_train_raw, y_train_raw, epochs=1, batch_size=10) #, validation_data=(x_valid, y_valid))

testLoss, testAcc = model.evaluate(x_test_raw, y_test_raw)
print('Predicted test accuracy: ', testAcc)
print('Predicted test loss: ', testLoss)

# Currently bugged
# nn.NeuralNets.metrics_generation(model, x_test, y_test)
