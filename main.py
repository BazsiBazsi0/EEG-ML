import numpy as np
import os
import loader, nn, generator

#generator.Generator.generate()

# Required data: how many patients t load
patients_to_load = 10
x, y, x_no_smote, y_no_smote = loader.FileLoader.load_saved_files(patients=patients_to_load)
x_no_rest, y_no_rest = loader.FileLoader.data_equalizer(x_no_smote, y_no_smote)
print('Shape of x: ', np.shape(x))

# TODO make runs with non-smote + balanced data (reduce rest) + keep test balanced
    #   The rest remove is done
    #   Next is modifying it or writing a fucntion that also measures the other instances
# TODO try to compare the data with oversampling the non-rest, try the same for test
# TODO Confusion matrix generation
# TODO szakdoga sablon methods: somte stb, jövőhétre vázlat, desktop beállítás
# SMOTE comparison, sums up per patients, then sums up everything

print('Instances of classes no SMOTE: ', y_no_smote.sum(axis=1).sum(axis=0))
print('Instances of classes data eq: ', y_no_rest.sum(axis=1).sum(axis=0))
print('Instances of classes after SMOTE: ', y.sum(axis=1).sum(axis=0))

models, history_loo, acc, avgAcc = nn.NeuralNets.leave_one_out(x_no_smote, y_no_smote)

print('Final avg accuracy:', avgAcc)

nn.NeuralNets.metrics_csv('accuracy.csv', acc, avgAcc)




