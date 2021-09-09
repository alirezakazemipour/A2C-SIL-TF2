from comet_ml import Experiment
from Common import Worker, Play, Logger, get_params
import multiprocessing as mp
import numpy as np
from Brain import Brain
import gym
from tqdm import tqdm
import time


if __name__ == '__main__':
    params = get_params()
    test_env = gym.make(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})
    test_env.close()
    del test_env
    params.update({"n_workers": mp.cpu_count()})
    params.update({"rollout_length": 80 // params["n_workers"]})

    brain = Brain(**params)
    if not params["do_test"]:
        experiment = Experiment() # Add your Comet configs!
        logger = Logger(brain, experiment=experiment, **params)

        if not params["train_from_scratch"]:
            init_iteration, episode = logger.load_weights()
        else:
            init_iteration = 0
            episode = 0

        mp.set_start_method("spawn")
        parents = []
        for i in range(params["n_workers"]):
            parent_conn, child_conn = mp.Pipe()
            parents.append(parent_conn)
            w = Worker(i, conn=child_conn, **params)
            w.start()

        rollout_base_shape = params["n_workers"], params["rollout_length"]

        total_states = np.zeros(rollout_base_shape + params["state_shape"], dtype=np.uint8)
        total_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
        total_rewards = np.zeros(rollout_base_shape)
        total_dones = np.zeros(rollout_base_shape, dtype=np.bool)
        total_values = np.zeros(rollout_base_shape)
        next_states = np.zeros((rollout_base_shape[0],) + params["state_shape"], dtype=np.uint8)

        logger.on()
        episode_reward = 0
        episode_length = 0
        for iteration in tqdm(range(init_iteration + 1, params["total_iterations"] + 1)):
            start_time = time.time()

            for t in range(params["rollout_length"]):
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

                episode_reward += total_rewards[0, t]
                episode_length += 1
                if total_dones[0, t]:
                    episode += 1
                    logger.log_episode(episode, episode_reward, episode_length)
                    episode_reward = 0
                    episode_length = 0

            _, next_values = brain.get_actions_and_values(next_states, batch=True)

            total_states = np.concatenate(total_states)
            total_actions = np.concatenate(total_actions)

            training_logs = brain.train(total_states,
                                        total_actions,
                                        total_rewards,
                                        total_dones,
                                        total_values,
                                        next_values)

            logger.log_iteration(iteration, training_logs)


    else:
        play = Play(params["env_name"], brain)
        play.evaluate()
