
# EEG-ML
![banner](assets/banner.webp)
<h2 align="center">EEG Signal Classification with neural networks for BMI Applications</h2>

<p align="center">
<a href=""><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
<a href=""><img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"></a>
</p>
<p align="center">
<a href="https://codecov.io/gh/bkutasi/EEG-ML" > 
 <img src="https://codecov.io/gh/bkutasi/EEG-ML/graph/badge.svg?token=5ZH3RH6PF9"/> 
 </a>
</p>



## Introduction

This project uses machine learning techniques to analyze and classify EEG Motor Movement/Imagery Dataset from PhysioNet. This code was initially written as a part of my MSc thesis and later improved while keeping the initial logic and idea the same.
  
All important information is in the [Documentation](https://bkutasi.github.io/EEG-ML/)

## Use case

EEG classification can be utilized to infer and classify the imagined motor movements thus helping a subject to control a device though a Brain Machine Interface.

## Problem Statement

The EEG Motor Movement/Imagery Dataset is imbalanced, which can lead to a bias in the machine learning model towards the majority class. This project aims to enhance and balance the imbalanced classes using synthetic data upsampling to achieve higher accuracy. The code includes various methods of upsampling and model training and evaluations.

## Dataset

The dataset used in this project is the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) from PhysioNet. It contains EEG data from subjects performing motor/imagery tasks.

## Methodology

1. **Data Preprocessing**: The EEG data is preprocessed for the machine learning model.
2. **Synthetic Data Upsampling**: To address the issue of class imbalance, synthetic data upsampling is performed(SMOTE). This helps in balancing the classes and provides more data for the model to learn from.
3. **Model Training**: A machine learning model is trained using TensorFlow. The model is trained on the upsampled dataset.
4. **Model Evaluation**: The performance of the model is evaluated on a separate test set.

## Code Usage

For the first time running it will take some time to download the raw dataset and generate the filtered data.

To run the code use linux or WSL, and follow these steps:

<details>
<summary> Important note if having difficulties with TensorFlow!</summary>

There are many problems currently with the Tensorflow package, the easiest way running it with GPU acceleration is in a Colab or Kaggle notebook. Currently a demo notebook is on the way, once the code refactoring is finished. If you are lucky and everything is right you can run it in docker (after installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) or in a venv.

</details>

### Local
Make a virtual environment
```
python3 -m venv env
```
Install requirements

Install tensorflow
```
python3 -m pip install tensorflow[and-cuda]
```
```
pip install -r requirements.txt
```
Run the main script

```
python3 main.py
```

### Docker
1. Clone the repository.
2. Build the dockerfile.
```
docker build -t EEG-ML .
```
3. Run the docker image and mount your working folder
```
docker run --rm -it -p 8888:8888/tcp -v ${PWD}:/workspace eeg-ml
```
4. Run the main script.
```
python3 main.py
```
If you already have the dataset:
```
docker run --rm -it -p 8888:8888/tcp -v ${PWD}:/workspace -v ${PWD}/dataset:/workspace/dataset eeg-ml
```

This setup will:
- Mount your current directory to /workspace
- Mount your existing dataset folder to prevent redownloading
- Use GPU acceleration if available

## Results

The restults are explained in the [Documentation](https://bkutasi.github.io/EEG-ML/)

## Future Work

Future work includes exploring different upsampling techniques and machine learning models to further improve the performance. Currently I'm working on getting things into a better shape including visuals. Docker deployment and a simple showcase jupyter notebook is also on the todo list.

## TODO
- ✅ High percentage testing coverage, coverage report
- ✅ Classes better OOP structure
  - ✅ Modularity 
  - ✅ Expandable multi class structure
  - ✅ Static helpers
- ✅Assessment and improvement of the results
  - ✅Visual representation
  - ✅Statistical tests
- ✅ Wiki page with explanations
- ✅Docker refinement
- Demo notebook

## Notes about testing
- Due to version differences and overall discrepancy over the years/months with the python test discovery inside vscode, it is recommended to put the following inside your `settings.json` file:
```json
"python.experiments.optOutFrom": ["pythonTestAdapter"]
```
- The S001 folder is the first subject from the dataset and it is mandatory to test the functionality of the various classes. Unfortunately it increases the size of the repo.
