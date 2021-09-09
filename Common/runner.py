from Common import *
from multiprocessing import Process


class Worker(Process):
    def __init__(self, id, conn, **config):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
        self.env = make_atari(self.config["env_name"], episodic_life=False)
        self.conn = conn
        self.episode_buffer = []
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        state = np.zeros(self.config["state_shape"], dtype=np.uint8)
        return stack_states(state, obs, True)

    def run(self):
        print(f"W: {self.id} started.")
        state = self.reset()
        while True:
            self.conn.send(state)
            action, value = self.conn.recv()
            next_obs, r, done, info = self.env.step(action)
            next_state = stack_states(state, next_obs, False)
            reward = np.sign(r)
            self.conn.send((next_state, reward, done))
            self.episode_buffer.append((state, action, reward, done, next_state, value))
            state = next_state
            if done:
                self.conn.send(self.episode_buffer)
                self.episode_buffer = []
                state = self.reset()
