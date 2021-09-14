from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense,LSTMCell
from tensorflow_probability.python.distributions import Categorical


class NN(Model, ABC):
    #  https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py

    def __init__(self, n_actions):
        super(NN, self).__init__()
        self.n_actions = n_actions

        self.conv1 = Conv2D(filters=16, kernel_size=8, strides=4, activation="relu", kernel_initializer="he_normal")
        self.conv2 = Conv2D(filters=32, kernel_size=4, strides=2, activation="relu", kernel_initializer="he_normal")
        self.flatten = Flatten()
        self.lstm = LSTMCell(units=256)
        self.value = Dense(units=1)
        self.logits = Dense(units=self.n_actions)

    def call(self, inputs, training=None, mask=None):
        x, hx, cx = inputs
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        hx, cx = self.lstm(x, (hx, cx))
        value = self.value(hx)
        dist = Categorical(logits=self.logits(hx))
        return dist, value, cx[0], cx[1]
