import os
from runner import Worker
from multiprocessing import Pipe, set_start_method
import numpy as np
from brain import Brain
import gym
from tqdm import tqdm
import time
from test_policy import evaluate_policy
from play import Play

env_name = "PongNoFrameskip-v4"
test_env = gym.make(env_name)
n_actions = test_env.action_space.n
n_workers = os.cpu_count()
state_shape = (84, 84, 4)
iterations = int(2e4)
log_period = 10
T = 80 // n_workers
lr = 7e-4
LOAD_FROM_CKP = True
Train = True


if __name__ == '__main__':
    set_start_method("spawn")
    brain = Brain(state_shape, n_actions, n_workers, lr)
    if Train:
        if LOAD_FROM_CKP:
            running_reward, init_iteration = brain.load_params()
        else:
            init_iteration = 0
            running_reward = 0

        parents = []
        for i in range(n_workers):
            parent_conn, child_conn = Pipe()
            parents.append(parent_conn)
            w = Worker(i, state_shape, env_name, child_conn)
            w.start()

        for iteration in tqdm(range(init_iteration + 1, iterations + 1)):
            start_time = time.time()
            total_states = np.zeros((n_workers, T,) + state_shape)
            total_actions = np.zeros((n_workers, T))
            total_rewards = np.zeros((n_workers, T))
            total_dones = np.zeros((n_workers, T))
            total_values = np.zeros((n_workers, T))
            next_states = np.zeros((n_workers,) + state_shape)
            next_values = np.zeros(n_workers)

            for t in range(T):
                for worker_id, parent in enumerate(parents):
                    s = parent.recv()
                    total_states[worker_id, t] = s

                total_actions[:, t], total_values[:, t] = brain.get_actions_and_values(total_states[:, t], batch=True)

                for parent, a in zip(parents, total_actions[:, t]):
                    parent.send(int(a))

                for worker_id, parent in enumerate(parents):
                    s_, r, d = parent.recv()
                    total_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_
            _, next_values = brain.get_actions_and_values(next_states, batch=True)

            total_states = np.concatenate(total_states)
            total_actions = np.concatenate(total_actions)

            # Calculates if value function is a good predictor of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            total_loss, entropy, ev = brain.train(total_states, total_actions, total_rewards, total_dones, total_values,
                                                  next_values)

            episode_reward = evaluate_policy(env_name, brain, state_shape)

            if iteration == 1:
                running_reward = episode_reward
            else:
                running_reward = 0.99 * running_reward + 0.01 * episode_reward

            if iteration % log_period == 0:
                print(f"Iter: {iteration}| "
                      f"Ep_reward: {episode_reward:.3f}| "
                      f"Running_reward: {running_reward:.3f}| "
                      f"Total_loss: {total_loss:.3f}| "
                      f"Explained variance:{ev:.3f}| "
                      f"Entropy: {entropy:.3f}| "
                      f"Iter_duration: {time.time() - start_time:.3f}| ")
                brain.save_params(iteration, running_reward)

            # with tf.summary.SummaryWriter(env_name + "/logs") as writer:
            #     writer.add_scalar("running reward", running_reward, iteration)
            #     writer.add_scalar("episode reward", episode_reward, iteration)
            #     writer.add_scalar("explained variance", ev, iteration)
            #     writer.add_scalar("loss", total_loss, iteration)
            #     writer.add_scalar("entropy", entropy, iteration)
    else:
        play = Play(env_name, brain)
        play.evaluate()
