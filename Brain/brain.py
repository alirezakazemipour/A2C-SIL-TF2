from .model import NN
import numpy as np
from Common import explained_variance
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error


class Brain:
    def __init__(self, **config):
        self.config = config
        self.policy = NN(self.config["n_actions"])
        self.optimizer = Adam(learning_rate=self.config["lr"])

    @tf.function
    def feedforward_model(self, x):
        dist, value = self.policy(x)
        action = dist.sample()
        return action, value

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        a, v = self.feedforward_model(state)
        return a.numpy(), v.numpy().squeeze()

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones)
        values = np.hstack(values)
        advs = (returns - values).astype(np.float32)

        a_loss, v_loss, ent, g_norm = self.optimize(states, actions, returns, advs)

        return a_loss.numpy(), v_loss.numpy(), ent.numpy(), g_norm, explained_variance(values, returns)

    @tf.function
    def optimize(self, state, action, q_value, adv):
        with tf.GradientTape() as tape:
            dist, value = self.policy(state)
            entropy = tf.reduce_mean(dist.entropy())
            log_prob = dist.log_prob(action)
            actor_loss = -tf.reduce_mean(log_prob * adv)
            critic_loss = mean_squared_error(q_value, tf.squeeze(value, axis=-1))
            total_loss = actor_loss + self.config["critic_coeff"] * critic_loss - self.config["ent_coeff"] * entropy

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.config["max_grad_norm"])
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return actor_loss, critic_loss, entropy, grad_norm

    def get_returns(self, rewards, next_values, dones):

        returns = [[] for _ in range(self.config["n_workers"])]
        for worker in range(self.config["n_workers"]):
            R = next_values[worker]
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.config["gamma"] * R * (1 - dones[worker][step])
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")

