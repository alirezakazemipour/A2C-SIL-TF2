from comet_ml import Experiment
from Common import Worker, Play, Logger, get_params
from Brain import Brain
from tqdm import tqdm
from collections import namedtuple
import multiprocessing as mp
import numpy as np
import gym
import time
import os
import tensorflow as tf

if __name__ == '__main__':
    params = get_params()
    os.environ["PYTHONHASHSEED"] = str(params["seed"])
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    test_env = gym.make(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})
    test_env.close()
    del test_env
    params.update({"rollout_length": 80 // params["n_workers"]})
    params.update({"transition": namedtuple('Transition', ('state', 'action', 'reward', 'done', 'value', 'hx', 'cx'))})
    params.update({"final_annealing_beta_steps": params["total_iterations"] // 8})

    brain = Brain(**params)
    if not params["do_test"]:
        experiment = Experiment()  # Add your Comet configs!
        logger = Logger(brain, experiment=experiment, **params)

        if not params["train_from_scratch"]:
            init_iteration, episode = logger.load_weights()
        else:
            init_iteration = 0
            episode = 0

        parents = []
        for i in range(params["n_workers"]):
            parent_conn, child_conn = mp.Pipe()
            parents.append(parent_conn)
            w = Worker(i, conn=child_conn, **params)
            w.start()

        rollout_base_shape = params["n_workers"], params["rollout_length"]

        total_states = np.zeros(rollout_base_shape + params["state_shape"], dtype=np.uint8)
        total_hxs = np.zeros(rollout_base_shape + (256,), dtype=np.float32)
        total_cxs = np.zeros(rollout_base_shape + (256,), dtype=np.float32)
        total_actions = np.zeros(rollout_base_shape, dtype=np.int32)
        total_rewards = np.zeros(rollout_base_shape)
        total_dones = np.zeros(rollout_base_shape, dtype=np.bool)
        total_values = np.zeros(rollout_base_shape, dtype=np.float32)
        next_states = np.zeros((rollout_base_shape[0],) + params["state_shape"], dtype=np.uint8)
        next_hxs = np.zeros((rollout_base_shape[0],) + (256,), dtype=np.uint8)
        next_cxs = np.zeros((rollout_base_shape[0],) + (256,), dtype=np.uint8)

        logger.on()
        episode_reward = 0
        episode_length = 0
        for iteration in tqdm(range(init_iteration + 1, params["total_iterations"] + 1)):
            start_time = time.time()

            for t in range(params["rollout_length"]):
                for worker_id, parent in enumerate(parents):
                    s, hx, cx = parent.recv()
                    total_states[worker_id, t] = s
                    total_hxs[worker_id, t] = hx
                    total_hxs[worker_id, t] = cx

                total_actions[:, t], total_values[:, t], next_hxs, next_cxs = \
                    brain.get_actions_and_values(total_states[:, t], total_hxs[:, t], total_cxs[:, t], batch=True)

                for parent, a, v, hx, cx in zip(parents, total_actions[:, t], total_values[:, t], next_hxs[:],
                                                next_cxs[:]):
                    parent.send((int(a), v, hx, cx))

                for worker_id, parent in enumerate(parents):
                    s_, r, d = parent.recv()
                    total_rewards[worker_id, t] = r
                    total_dones[worker_id, t] = d
                    next_states[worker_id] = s_

                for parent, done in zip(parents, total_dones[:, t]):
                    if done:
                        brain.add_to_memory(*parent.recv())

                episode_reward += total_rewards[0, t]
                episode_length += 1
                if total_dones[0, t]:
                    episode += 1
                    logger.log_episode(episode, episode_reward, episode_length)
                    episode_reward = 0
                    episode_length = 0

            _, next_values, *_ = brain.get_actions_and_values(next_states, next_hxs, next_cxs, batch=True)

            training_logs = brain.train_a2c(np.concatenate(total_states),
                                            np.concatenate(total_hxs),
                                            np.concatenate(total_cxs),
                                            np.concatenate(total_actions),
                                            np.sign(total_rewards),
                                            total_dones,
                                            total_values,
                                            next_values)

            beta = min(1.0, params["beta"] + iteration * (1.0 - params["beta"]) / params["final_annealing_beta_steps"])
            for m in range(params["n_sil_updates"]):
                sil_training_logs = brain.train_sil(beta)

            logger.log_iteration(iteration, beta, training_logs)

    else:
        logger = Logger(brain, experiment=None, **params)
        logger.load_weights()
        play = Play(params["env_name"], brain)
        play.evaluate()
