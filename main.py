import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
import loader, nn, generator

#generator.Generator.generate()

x, y = loader.FileLoader.load_saved_files()
print('Shape of x: ', np.shape(x))

# Transform y to one-hot-encoding
y_one_hot = nn.NeuralNets.to_one_hot(y)
# Reshape for scaling
x_reshaped = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
# TODO train on 9 person, test on 1 person, make generation to only generate 160 epoch per person ###DONE
#   leave one person(160 epochs) out per iteration, every iteration different person is the test set ###DONE
#   Shuffle before slicing the arrays ###DONE implemented train_test_split with shuffle

x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(x_reshaped, y_one_hot, test_size=0.1, shuffle = True)

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

"""models = {'SMOTE': nn.NeuralNets.starter_net(),
          'NO_SMOTE': nn.NeuralNets.starter_net(),
          'leave_one_out': nn.NeuralNets.starter_net()}"""

history_loo = []
acc = []
avgAcc = []
for i in range(0, 3360, 160):
    print('Current loop index: ', i//160)
    loo_left_out_x = x_train[i:i+160]
    loo_left_out_y = y_train_raw[i:i+160]
    loo_left_in_x = np.append(x_train[:i], x_train[i+160:], 0)
    loo_left_in_y = np.append(y_train[:i], y_train[i+160:], 0)

    model = nn.NeuralNets.starter_net()
    history = model.fit(loo_left_in_x, loo_left_in_y,
                        epochs=5, batch_size=100,
                        validation_data=(loo_left_out_x, loo_left_out_y))
    history_loo.append(history)
    testLoss, testAcc = model.evaluate(loo_left_out_x, loo_left_out_y)
    # del model
    acc.append(testAcc)
    avgAcc = np.average(acc)
    print('Current avg acc:', avgAcc)

print('Final avg acc:', avgAcc)
# Final avg acc: 1.0 , LOO cv-d on 10 patients, 50 epochs, batch_size=25 acc across 10 loops

"""
history_smote = models['SMOTE'].fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_test, y_test_raw))
history_no_smote = models['NO_SMOTE'].fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_test, y_test_raw))

testLoss, testAcc = models['NO_SMOTE'].evaluate(x_test, y_test_raw)
print('Predicted test accuracy: ', testAcc)
print('Predicted test loss: ', testLoss)
"""


"""nn.NeuralNets.save_accuracy_curves(history_smote, 10, 'smote.png')
nn.NeuralNets.save_accuracy_curves(history_no_smote, 10, 'no_smote.png')

nn.NeuralNets.metrics_generation(models['SMOTE'], x_test, y_test_raw)
nn.NeuralNets.metrics_generation(models['NO_SMOTE'], x_test, y_test_raw)"""




