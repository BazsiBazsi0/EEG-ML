## Introduction

This project uses machine learning techniques to analyze and classify EEG Motor Movement/Imagery Dataset from PhysioNet. This code was initially written as a part of my MSc thesis and later improved while keeping the initial logic and idea the same.

::: danger I made some changes to my original code, so the results may differ from the original ones. Everything is documented in [Important Changes](/getting-started/important-changes) 
:::

## Use case

EEG classification can be utilized to infer and classify the imagined motor movements thus helping a subject to control a device though a Brain Machine Interface.

## Problem Statement

The EEG Motor Movement/Imagery Dataset is imbalanced, which can lead to a bias in the machine learning model towards the majority class. This project aims to enhance and balance the imbalanced classes using synthetic data upsampling to achieve higher accuracy. The code includes various methods of upsampling and model training and evaluations.
## Dataset

The EEG Motor Movement/Imagery Dataset from PhysioNet consists of EEG recordings from subjects performing motor movements or imagining motor movements. The dataset is labeled with the corresponding motor movement class. The dataset is imbalanced, with some classes having significantly fewer samples than others.

## Approach

To address the class imbalance, this project utilizes synthetic data upsampling techniques. Up- and downsampling methods, such as SMOTE is implemented to generate synthetic samples for the minority classes. The upsampling is performed prior to model training to ensure a balanced representation of all classes.

## Model Training and Evaluation

After upsampling the dataset, machine learning models are trained using various algorithms, such as Common Spacial Patterns, XDAWN and Neural Networks. The models are evaluated using metrics such as accuracy, precision, recall, and F1-score to assess their performance in classifying the EEG motor movement/imagery data.
