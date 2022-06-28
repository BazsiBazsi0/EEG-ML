import numpy as np
import os
import loader, nn, generator

#generator.Generator.generate()

# Required data: how many patients t load
patients_to_load = 10
test_size = 0.1
x, y, x_no_smote, y_no_smote = loader.FileLoader.load_saved_files(patients=patients_to_load)
print('Shape of x: ', np.shape(x))

# SMOTE comparison, sums up per patients, then sums up everything
print('Instances of classes before SMOTE: ', y_no_smote.sum(axis=1).sum(axis=0))
print('Instances of classes after SMOTE: ', y.sum(axis=1).sum(axis=0))

history_loo = []
acc = []
avgAcc = []
for i in range(0, patients_to_load):
    print('Current loop index: ', i)
    # Leaving out one person from the training
    x_loop = np.append(x_no_smote[:i], x_no_smote[i+1:], 0)
    # Reshaping into 2D array
    x_2d = x_loop.reshape((int(x_loop.shape[0] * x_loop.shape[1]), x_loop.shape[2], x_loop.shape[3]))

    y_loop = np.append(y_no_smote[:i], y_no_smote[i+1:], 0)
    y_2d = y_loop.reshape((int(y_loop.shape[0] * y_loop.shape[1]), y_loop.shape[2]))

    x_val = x_no_smote[i]
    y_val = y_no_smote[i]

    model = nn.NeuralNets.basic_net()

    # Validation is done on person left out on from the training from the train set
    history = model.fit(x_2d, y_2d, epochs=10, batch_size=100, validation_data=(x_val, y_val))
    history_loo.append(history)

    # Saving and averaging the accuracies, evaluation is done on the original dataset
    testLoss, testAcc = model.evaluate(x_val, y_val)
    acc.append(testAcc)
    avgAcc = np.average(acc)
    print('Current avg acc:', avgAcc)

#np.save(os.path.join(os.getcwd(), "accuracy"), acc, allow_pickle=True)
np.savetxt("accuracy_no_smote.csv", acc, delimiter=",", fmt = '%.4f')
print('Final avg acc:', avgAcc)
# Trained on SMOTE:
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




