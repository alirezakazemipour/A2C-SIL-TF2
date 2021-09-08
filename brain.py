from model import NN
import numpy as np
from utils import explained_variance
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json


class Brain:
    def __init__(self, state_shape, n_actions, n_workers, lr):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.n_workers = n_workers
        self.lr = lr

        self.current_policy = NN(self.n_actions)
        self.optimizer = Adam(learning_rate=self.lr, epsilon=1e-5)

    @tf.function
    def feedforward_model(self, x):
        dist, value = self.current_policy(x)
        action = dist.sample()
        return action, value

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        a, v = self.feedforward_model(state)
        return a.numpy(), v.numpy().squeeze()

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones).astype("float32")
        values = np.vstack(values)
        advs = returns - values
        advs = (advs - advs.mean(1).reshape((-1, 1))) / (advs.std(1).reshape((-1, 1)) + 1e-6)

        total_loss, entropy = self.optimize(states, actions, np.hstack(returns), np.hstack(advs).astype("float32"))

        return total_loss.numpy(), entropy.numpy(), explained_variance(np.hstack(values), np.hstack(returns))

    @tf.function
    def optimize(self, state, action, q_value, adv):
        with tf.GradientTape() as tape:
            dist, value = self.current_policy(state)
            entropy = tf.reduce_mean(dist.entropy())
            new_log_prob = dist.log_prob(action)
            actor_loss = -tf.reduce_mean(new_log_prob * adv)

            critic_loss = tf.reduce_mean(0.5 * (q_value - value) ** 2)

            total_loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

        grads = tape.gradient(total_loss, self.current_policy.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.current_policy.trainable_variables))

        return total_loss, entropy

    def get_returns(self, rewards, next_values, dones, gamma=0.99):

        returns = [[] for _ in range(self.n_workers)]
        for worker in range(self.n_workers):
            R = next_values[worker] if not isinstance(next_values, float) else next_values
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + gamma * R * (1 - dones[worker][step])
                returns[worker].insert(0, R)

        return np.vstack(returns)

    def save_params(self, iteration, running_reward):
        self.current_policy.save_weights("weights.h5", save_format="h5")
        stats_to_write = {"iteration": iteration, "running_reward": running_reward}
        with open("stats.json", "w") as f:
            f.write(json.dumps(stats_to_write))
            f.flush()

    def load_params(self):
        self.current_policy.build((None, *self.state_shape))
        self.current_policy.load_weights("weights.h5")
        with open("stats.json", "r") as f:
            stats = json.load(f)

        return stats["running_reward"], stats["iteration"]
