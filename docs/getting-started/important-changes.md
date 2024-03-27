There are some important changes that were made to the original code. The most important ones are:
- The code was refactored to be more modular and easier to understand.
- The code was updated to use the latest version of TensorFlow and Keras.
- The processing is refined and the code is optimized for better performance and understandability.
- The final dimensions of the model are changed to be more suitable for the data, swapped from epochs x channels x samples to epochs x samples x channels.
- Originally I used 4 second from the 5 second epochs, but I changed it to include all the five seconds of the epochs. This lead to a significant improvement in classification scores.

Problems i encountered and recommendations:
- While the neural network structure is sufficiently complex, the model cannot learn to classify the data until perfection. This is likely due to the small size of the dataset and the complexity of the features.
- When applying SMOTE the model doesn't learn the features of the dataset better, it just learns the synthetic data.
    - This can be explain by the fact that the synthetic data is not representative of the real data.
    - The model learns whats similar and frequent between examples and not the "features" of the data.
    - The introduction of SMOTE upsampling introduced gaussian noise to the data, which is not present in the real data.
    - The model therefore learned the gaussian noise which was present in all of the synthetic data, which resulted very high accuracy on the synthetic data but very low accuracy on the real data.
    - Recommendation about SMOTE upsampling: It should be used with caution and the results should be validated on a separate test set.
    - It is also recommended to not to use too much upsampling and only introduce the method when high enough accuracy is not achieved with the real data.
    - Here are some recommendations to improve your project:
        - You can equalize the classes with the equalizer function i wrote in the FileProcessor class.
        - You can also remove the majority class to reduce the over-representation of the majority class.
        - Always inspect the classification report for the model to see the performance on the real data, because accuracy is nor a reliable metric for multiclass classification.
        - You can try building a single class classifier.
        - Try playing around with frequencies, preprocessing, ROIs(1-2 s reduced "cut" epochs), and other features.
        - If this is your first time working with EEG data, try building simple classification pipelines with the raw data, and try to understand the data and the features. You can also try learning more about image classification since images are also signals but you can have a way better understanding of the features.