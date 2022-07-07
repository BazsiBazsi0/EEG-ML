import numpy as np
import os
import loader, nn, generator

#generator.Generator.generate()

# Required data: how many patients t load
patients_to_load = 10
x, y, x_no_smote, y_no_smote = loader.FileLoader.load_saved_files(patients=patients_to_load)
print('Shape of x: ', np.shape(x))

# SMOTE comparison, sums up per patients, then sums up everything
print('Instances of classes before SMOTE: ', y_no_smote.sum(axis=1).sum(axis=0))
print('Instances of classes after SMOTE: ', y.sum(axis=1).sum(axis=0))

models, history_loo, acc, avgAcc = nn.NeuralNets.leave_one_out(x_no_smote, y_no_smote)

print('Final avg accuracy:', avgAcc)

nn.NeuralNets.metrics_csv('accuracy.csv', acc, avgAcc)





