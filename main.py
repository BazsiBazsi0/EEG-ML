import numpy as np
import loader
import nn
from utils.dataset_download_utils import Downloader
from utils.dataset_utils import DatasetUtils

if __name__ == "__main__":
    downloader = Downloader(
        url="https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip"
    )

    downloader.download()

    dataset_utils = DatasetUtils()
    dataset_utils.generate()


    
    x, y = loader.FileLoader.load_saved_files()
    # x_fft, y_fft = nn.NeuralNets.fft_processor(x_no_smote, y_no_smote)
    # x_no_rest, y_no_rest = loader.FileLoader.data_equalizer(x_no_smote, y_no_smote)
    print("Shape of x: ", np.shape(x))

    # print('Instances of classes no SMOTE: ', y_no_smote.sum(axis=1).sum(axis=0))
    # print('Instances of classes data eq: ', y_no_rest.sum(axis=1).sum(axis=0))
    print("Instances of classes after SMOTE: ", y.sum(axis=1).sum(axis=0))

    history, model, acc, avgAcc = nn.NeuralNets.loo(x, y)

    # model = nn.NeuralNets.one_d_cnn_multi()
    # nn.NeuralNets.generator_processor(model, x_no_smote, y_no_smote)
    # models, history_loo, acc, avgAcc = nn.NeuralNets.leave_one_out(x_no_smote, y_no_smote)

    with open("avg accuracy.txt", "a") as f:
        f.write(str(avgAcc) + "\n")

    print("Final avg accuracy:", avgAcc)

    x, y, x_no_smote, y_no_smote = loader.FileLoader.load_saved_files_val()
    nn.NeuralNets.metrics_generation(model, x_no_smote, y_no_smote, "1DCNN")
    nn.NeuralNets.plot_roc_curve(model, x_no_smote, y_no_smote, "1DCNN")

    """nn.NeuralNets.metrics_generation(models['FCN'], x_no_smote, y_no_smote)
    nn.NeuralNets.plot_roc_curve(models['FCN'], x_no_smote, y_no_smote, "FCN.png")"""
