import tensorflow as tf
import numpy as np
from neuralnets.training_utils.GradAugAdam import GradAugAdam


class DoublePredictor:
    """
    A class that predicts using two GPUs
    At the moment it doenst work, it cant load the weights
    """

    def __init__(self, model):
        self.model = model
        self.model1 = tf.keras.models.clone_model(model)
        self.model1.set_weights(model.get_weights())
        self.model2 = tf.keras.models.clone_model(model)
        self.model2.set_weights(model.get_weights())

    def predict(self, data):
        data1, data2 = np.array_split(data, 2)

        with tf.device("/GPU:0"):
            predictions1 = self.model1.predict(data1)

        with tf.device("/GPU:1"):
            predictions2 = self.model2.predict(data2)

        predictions = np.concatenate([predictions1, predictions2])
        return predictions
