from .model import NN
from .experience_replay import ReplayMemory
from Common import explained_variance
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


class Brain:
    def __init__(self, **config):
        self.config = config
        tf.random.set_seed(self.config["seed"])
        self.policy = NN(self.config["n_actions"])
        self.policy.build(input_shape=[(None, *self.config["state_shape"]), (None, 256), (None, 256)])
        self.optimizer = Adam(learning_rate=self.config["lr"])
        self.memory = ReplayMemory(self.config["mem_size"], self.config["alpha"], seed=self.config["seed"])

    def extract_rewards(self, *x):
        batch = self.config["transition"](*zip(*x))
        rewards = np.array(batch.reward).reshape(1, -1)
        dones = np.array(batch.done).reshape(1, -1)
        return rewards, dones

    def unpack_batch(self, x):
        batch = self.memory.transition(*zip(*x))

        states = np.concatenate(batch.s).reshape(-1, *self.config["state_shape"])
        hxs = np.vstack(batch.hx)
        cxs = np.vstack(batch.cx)
        actions = np.array(batch.a)
        returns = np.array(batch.R, dtype=np.float32)
        advs = np.array(batch.adv, dtype=np.float32)
        return states, hxs, cxs, actions, returns, advs

    def add_to_memory(self, *trajectory):
        rewards, dones = self.extract_rewards(*trajectory)
        returns = self.get_returns(rewards, np.asarray(0), dones, 1)
        for transition, R in zip(trajectory, returns):
            s, a, *_, v, hx, cx = transition
            self.memory.add(s, hx, cx, a, R, R - v)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 84, 84, 4), dtype=tf.uint8),
                                  tf.TensorSpec(shape=(None, 256), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 256), dtype=tf.float32)
                                  ]
                 )
    def feedforward_model(self, x, hx, cx):
        dist, value, hx, cx = self.policy((x, hx, cx))
        # action = dist.sample()
        # print(value)
        return dist.logits, value, hx, cx

    def get_actions_and_values(self, state, hx, cx, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        logits, v, hx, cx = self.feedforward_model(tf.constant(state),
                                                   tf.constant(hx, dtype=tf.float32),
                                                   tf.constant(cx, dtype=tf.float32)
                                                   )
        action = tf.random.categorical(logits, num_samples=1)
        return action.numpy().squeeze(), v.numpy().squeeze(), hx.numpy(), cx.numpy()

    def train(self, states, hxs, cxs, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones, n=self.config["n_workers"])
        values = np.hstack(values)
        advs = (returns - values).astype(np.float32)

        a_loss, v_loss, ent, g_norm, grads = self.optimize(tf.constant(states),
                                                    tf.constant(hxs, dtype=tf.float32),
                                                    tf.constant(cxs, dtype=tf.float32),
                                                    tf.constant(actions),
                                                    tf.constant(returns),
                                                    tf.constant(advs),
                                                    weights=tf.constant(np.ones(actions.shape[0]), dtype=tf.float32),
                                                    masks=tf.constant(np.ones(actions.shape[0]), dtype=tf.float32),
                                                    batch_size=tf.constant(
                                                        self.config["rollout_length"] * self.config["n_workers"],
                                                        dtype=tf.float32),
                                                    critic_coeff=tf.constant(self.config["critic_coeff"],
                                                                             dtype=tf.float32),
                                                    ent_coeff=tf.constant(self.config["ent_coeff"], dtype=tf.float32)
                                                    )
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        return a_loss.numpy(), v_loss.numpy(), ent.numpy(), g_norm.numpy(), explained_variance(values, returns)

    def train_sil(self, beta):
        if len(self.memory) < self.config["sil_batch_size"]:
            return 0, 0, 0, 0
        batch, weights, indices = self.memory.sample(self.config["sil_batch_size"], beta)
        states, hxs, cxs, actions, returns, advs = self.unpack_batch(batch)
        masks = (advs >= 0).astype("float32")
        batch_size = np.sum(masks)
        if batch_size != 0:
            a_loss, v_loss, ent, g_norm, grads = self.optimize(tf.constant(states),
                                                        tf.constant(hxs, dtype=tf.float32),
                                                        tf.constant(cxs, dtype=tf.float32),
                                                        tf.constant(actions),
                                                        tf.constant(returns),
                                                        tf.constant(advs * masks),
                                                        weights=tf.constant(weights),
                                                        masks=tf.constant(masks),
                                                        batch_size=tf.constant(batch_size, dtype=tf.float32),
                                                        critic_coeff=tf.constant(self.config["w_vloss"],
                                                                                 dtype=tf.float32),
                                                        ent_coeff=tf.constant(0, dtype=tf.float32)
                                                        )
            self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
            self.memory.update_priorities(indices, advs * masks + 1e-6)
            return a_loss.numpy(), v_loss.numpy(), ent.numpy(), g_norm.numpy()

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 84, 84, 4), dtype=tf.uint8),
                                  tf.TensorSpec(shape=(None, 256), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 256), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.int64),
                                  tf.TensorSpec(shape=(None,), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None,), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.float32)
                                  ]
                 )
    def optimize(self, state, hx, cx, action, q_value, adv, weights, masks, batch_size, critic_coeff, ent_coeff):
        # print(batch_size)
        with tf.GradientTape() as tape:
            dist, value, *_ = self.policy((state, hx, cx))
            entropy = tf.reduce_sum(dist.entropy() * weights * masks) / batch_size
            log_prob = dist.log_prob(action)
            actor_loss = -tf.reduce_sum(log_prob * adv * weights) / batch_size
            critic_loss = tf.reduce_sum(
                0.5 * tf.square(q_value - tf.squeeze(value, axis=-1)) * weights * masks) / batch_size
            total_loss = actor_loss + critic_coeff * critic_loss - ent_coeff * entropy

        grads = tape.gradient(total_loss, self.policy.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, self.config["max_grad_norm"])

        return actor_loss, critic_loss, entropy, grad_norm, grads

    def get_returns(self, rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, n: int) -> np.ndarray:
        if next_values.shape == ():
            next_values = next_values[None]

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.config["gamma"] * R * (1 - dones[worker][step])
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")
