from Common import *
from multiprocessing import Process


class Worker(Process):
    def __init__(self, id, conn, **config):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
        self.env = make_atari(self.config["env_name"], episodic_life=False, seed=self.config["seed"] + self.id)
        self.conn = conn
        self.reward = 0
        self.episode_buffer = []
        self.sign = lambda x: bool(x > 0) - bool(x < 0)
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
        print(f"W{self.id}: started.")
        state = self.reset()
        hx = np.zeros((1, 256), dtype=np.float32)
        cx = np.zeros((1, 256), dtype=np.float32)
        while True:
            self.conn.send((state, hx, cx))
            action, value, next_hx, next_cx = self.conn.recv()
            next_obs, reward, done, info = self.env.step(action)
            # self.render()
            if reward > self.reward:
                self.reward = reward
                print("\n🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁")
                print(f"W{self.id} got a reward!!!")
                print("🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁🎁")
            next_state = stack_states(state, next_obs, False)
            self.conn.send((next_state, reward, done))
            self.episode_buffer.append((state, action, self.sign(reward), done, value, hx, cx))
            state = next_state
            hx = next_hx
            cx = next_cx
            if done:
                self.conn.send(self.episode_buffer)
                self.episode_buffer = []
                state = self.reset()
                hx = np.zeros((1, 256), dtype=np.float32)
                cx = np.zeros((1, 256), dtype=np.float32)
