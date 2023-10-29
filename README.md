
# EEG-ML
![banner](assets/banner.webp)
<h2 align="center">EEG Signal Classification with neural networks for BMI Applications</h2>

<p align="center">
<a href=""><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a>
<a href=""><img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"></a>
</p>

## Introduction

This project uses machine learning techniques to analyze and classify EEG Motor Movement/Imagery Dataset from PhysioNet. This code was initially written as a part of my MSc thesis and later improved while keeping the initial logic and idea the same.

## Use case

EEG classification can be utilized to infer and classify the imagined motor movements thus helping a subject to control a device though a Brain Machine Interface.

## Problem Statement

The EEG Motor Movement/Imagery Dataset is imbalanced, which can lead to a bias in the machine learning model towards the majority class. This project aims to enhance and balance the imbalanced classes using synthetic data upsampling to achieve higher accuracy. The code includes various methods of upsampling and model training and evaluations.

## Dataset

The dataset used in this project is the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) from PhysioNet. It contains EEG data from subjects performing motor/imagery tasks.

## Methodology

1. **Data Preprocessing**: The EEG data is preprocessed to remove noise and make it suitable for the machine learning model.
2. **Synthetic Data Upsampling**: To address the issue of class imbalance, synthetic data upsampling is performed(SMOTE). This helps in balancing the classes and provides more data for the model to learn from.
3. **Model Training**: A machine learning model is trained using TensorFlow. The model is trained on the upsampled dataset.
4. **Model Evaluation**: The performance of the model is evaluated on a separate test set.

## Results
The individual results are in the results folder in their respective folder.


## Code Usage
For the first time running its gonna take some time to download the raw dataset and generate the filtered data.

To run the code use linux or WSL, and follow these steps:

0. pray

There are many problems currently with the Tensorflow package, the easiest way running it with GPU acceleration is in a Colab or Kaggle notebook. Currently a demo notebook is on the way, once the code refactoring is finished. If you are lucky and everything is right you can run it in docker (after installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) or in a venv.

# Docker
1. Clone the repository.
2. Build the dockerfile.
```
sudo docker build -t EEG-ML .
```
1. Run the docker image and mount your working folder
```
docker run --rm -it -p 8888:8888/tcp -v ${PWD}:/workspace EEG-ML:latest
```
1. Run the main script.
```
python3 main.py
```

# Local
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

> Note if you want to run it on VSCode, make sure to use the _ms-python.python-2023.8.0_ extension. Testing is currently only supported this way.

## Results

The use of synthetic data upsampling significantly improved the performance of the machine learning model on the imbalanced EEG Motor Movement/Imagery Dataset. The individual results are in the "Results" folder.

## Future Work

Future work includes exploring different upsampling techniques and machine learning models to further improve the performance. Currently I'm working on getting things into a better shape including visuals. Docker deployment and a simple showcase jupyter notebook is also on the todo list.

## TODO
- High percentage testing coverage, coverage report
    - utils test created, basic utils are covered
    - test coverage report WIP
- Demo notebook
- Classes
  - preprocessing and core logic: WIP
  - dataset generation DONE
- Modularity and expandable multi class structure
  - utils DONE - needs review
