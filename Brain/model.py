from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow_probability.python.distributions import Categorical
import tensorflow as tf


class NN(Model, ABC):
    #  https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py

    def __init__(self, n_actions):
        super(NN, self).__init__()
        self.n_actions = n_actions

        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", kernel_initializer="he_normal")
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu", kernel_initializer="he_normal")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", kernel_initializer="he_normal")
        self.flatten = Flatten()
        self.fc = Dense(units=512, activation="relu", kernel_initializer="he_normal")
        self.value = Dense(units=1)
        self.logits = Dense(units=self.n_actions)

    def call(self, x, training=None, mask=None):
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        value = self.value(x)
        pi = tf.nn.softmax(self.logits(x))
        dist = Categorical(probs=pi)
        return dist, value
