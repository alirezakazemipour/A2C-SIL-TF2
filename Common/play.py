import time
from .utils import *


class Play:
    def __init__(self, env, agent, max_episode=5):
        self.env = make_atari(env, episodic_life=False)
        self.env = gym.wrappers.Monitor(self.env, "./Vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent

    def evaluate(self):
        stacked_states = np.zeros((84, 84, 4), dtype=np.uint8)
        mean_ep_reward = []
        for ep in range(self.max_episode):
            self.env.seed(ep)
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            clipped_ep_reward = 0
            hx, cx = np.zeros((1, 256), dtype=np.float32), np.zeros((1, 256), dtype=np.float32)
            for _ in range(self.env.spec.max_episode_steps):
                action, _, next_hx, next_cx = self.agent.get_actions_and_values(stacked_states, hx, cx)
                s_, r, done, info = self.env.step(action)
                episode_reward += r
                clipped_ep_reward += np.sign(r)
                if done :
                    break
                stacked_states = stack_states(stacked_states, s_, False)
                hx = next_hx
                cx = next_cx
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward}| "
                  f"clipped episode reward:{clipped_ep_reward}")
            mean_ep_reward.append(episode_reward)
            self.env.close()
        print(f"Mean episode reward:{sum(mean_ep_reward) / len(mean_ep_reward):0.2f}")
