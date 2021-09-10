from .model import NN
from .experience_replay import ReplayMemory
import numpy as np
from Common import explained_variance
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class Brain:
    def __init__(self, **config):
        self.config = config
        tf.random.set_seed(self.config["seed"])
        self.policy = NN(self.config["n_actions"])
        self.optimizer = Adam(learning_rate=self.config["lr"])
        self.memory = ReplayMemory(self.config["mem_size"], self.config["alpha"], seed=self.config["seed"])

    def extract_batch(self, *x):
        batch = self.config["transition"](*zip(*x))
        states = np.concatenate(batch.state).reshape(-1, *self.config["state_shape"])
        rewards = np.array(batch.reward).reshape(-1, 1)
        dones = np.array(batch.done).reshape(-1, 1)
        values = np.array(batch.value).reshape(-1, 1)
        return states, rewards, dones, values

    def unpack_batch(self, x):
        batch = self.memory.transition(*zip(*x))

        states = np.concatenate(batch.s).reshape(-1, *self.config["state_shape"])
        returns = np.concatenate(batch.R).astype(np.float32)
        actions = np.array(batch.a)
        advs = np.concatenate(batch.adv).astype(np.float32)
        return states, actions, returns, advs

    def add_to_memory(self, *trajectory):
        states, rewards, dones, values = self.extract_batch(*trajectory)
        returns = self.get_returns([rewards], [0], [dones], 1)
        for transition, R, V in zip(trajectory, returns, values):
            if R >= V:
                s, a, *_ = transition
                self.memory.add(s, a, R, R - V)

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
        returns = self.get_returns(rewards, next_values, dones, n=self.config["n_workers"])
        values = np.hstack(values)
        advs = (returns - values).astype(np.float32)

        a_loss, v_loss, ent, g_norm = self.optimize(states, actions, returns, advs)

        return a_loss.numpy(), v_loss.numpy(), ent.numpy(), g_norm.numpy(), explained_variance(values, returns)

    def train_sil(self, beta):
        if len(self.memory) < self.config["sil_batch_size"]:
            return 0, 0, 0, 0
        batch, weights, indices = self.memory.sample(self.config["sil_batch_size"], beta)
        states, actions, returns, advs = self.unpack_batch(batch)

        a_loss, v_loss, ent, g_norm = self.optimize(states,
                                                    actions,
                                                    returns,
                                                    advs,
                                                    weights=weights,
                                                    ent_coeff=0,
                                                    sil_beta=self.config["w_vloss"])
        self.memory.update_priorities(indices, advs + 1e-6)

        return a_loss.numpy(), v_loss.numpy(), ent.numpy(), g_norm.numpy()

    @tf.function
    def optimize(self, state, action, q_value, adv, weights=1, ent_coeff=0.01, sil_beta=1):
        with tf.GradientTape() as tape:
            dist, value = self.policy(state)
            entropy = tf.reduce_mean(dist.entropy() * weights)
            log_prob = dist.log_prob(action)
            actor_loss = -tf.reduce_mean(log_prob * adv * weights)
            critic_loss = tf.reduce_mean(tf.square(q_value - tf.squeeze(value, axis=-1)) * weights) * sil_beta
            total_loss = actor_loss + self.config["critic_coeff"] * critic_loss - ent_coeff * entropy

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.config["max_grad_norm"])
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return actor_loss, critic_loss, entropy, grad_norm

    def get_returns(self, rewards, next_values, dones, n):

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.config["gamma"] * R * (1 - dones[worker][step])
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")
