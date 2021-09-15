from abc import ABC
from tensorflow.keras.layers import Conv2D, Flatten, Dense,LSTMCell
from tensorflow_probability import distributions as tfd
import tensorflow as tf


class NN(tf.Module, ABC):
    #  https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py

    def __init__(self, n_actions):
        super(NN, self).__init__()
        self.n_actions = n_actions

        self.conv1 = Conv2D(filters=16, kernel_size=8, strides=4)
        self.conv2 = Conv2D(filters=32, kernel_size=4, strides=2)
        self.flatten = Flatten()
        self.lstm = LSTMCell(units=256)
        self.value = Dense(units=1, name="value_layer")
        self.logits = Dense(units=self.n_actions)

    def __call__(self, inputs, training=None, mask=None):
        x, hx, cx = inputs
        x = x / 255
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = self.flatten(x)
        hx, cx = self.lstm(x, (hx, cx))
        value = self.value(hx)
        dist = tfd.Categorical(logits=self.logits(hx))
        return dist, value, cx[0], cx[1]
