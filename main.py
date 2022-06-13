import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
import loader, nn, generator

#generator.Generator.generate()

x, y = loader.FileLoader.load_saved_files()

# Transform y to one-hot-encoding
y_one_hot = nn.NeuralNets.to_one_hot(y)
# Reshape for scaling
x_reshaped = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

x_train_raw = x_reshaped[:3000, :]
y_train_raw = y_one_hot[:3000, :]
x_test_raw = x_reshaped[3001:, :]
y_test_raw = y_one_hot[3001:, :]

# Scale between 0 and 1
x_train_scaled = minmax_scale(x_train_raw, axis=1)
x_test_scaled = minmax_scale(x_test_raw, axis=1)

# SMOTE
x_train_smote, y_train = nn.NeuralNets.smote_processor(x_train_scaled, y_train_raw)
print('classes count')
print('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print('after oversampling = {}'.format(y_train.sum(axis=0)))

# Transforms back into the original shape
x_train = x_train_smote.reshape(x_train_smote.shape[0], int(x_train_smote.shape[1] / 64), 64)
x_test = x_test_scaled.reshape(x_test_scaled.shape[0], int(x_test_scaled.shape[1] / 64), 64)
x_train_nosmote = x_train_scaled.reshape(x_train_scaled.shape[0], int(x_train_scaled.shape[1] / 64), 64)

loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()

models = {'SMOTE': nn.NeuralNets.starter_net(),
          'NO_SMOTE': nn.NeuralNets.starter_net()}

model = nn.NeuralNets.starter_net()
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

models['SMOTE'].compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
models['NO_SMOTE'].compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

history_smote = models['SMOTE'].fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_test, y_test_raw))
history_no_smote = models['NO_SMOTE'].fit(x_train_nosmote, y_train_raw, epochs=10, batch_size=100, validation_data=(x_test, y_test_raw))

"""testLoss, testAcc = models['NO_SMOTE'].evaluate(x_test, y_test_raw)
print('Predicted test accuracy: ', testAcc)
print('Predicted test loss: ', testLoss)"""

nn.NeuralNets.save_accuracy_curves(history_smote, 10, 'smote.png')
nn.NeuralNets.save_accuracy_curves(history_no_smote, 10, 'no_smote.png')

nn.NeuralNets.metrics_generation(models['NO_SMOTE'], x_test, y_test_raw)




