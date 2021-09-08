from utils import *
from multiprocessing import Process


class Worker(Process):
    def __init__(self, id, state_shape, env_name, conn):
        super(Worker, self).__init__()
        self.id = id
        self.env_name = env_name
        self.state_shape = state_shape
        self.env = make_atari(self.env_name)
        self.lives = self.env.ale.lives()
        self.conn = conn
        self._stacked_states = np.zeros(self.state_shape, dtype=np.uint8)
        self.reset()

    def __str__(self):
        return str(self.id)

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        self._stacked_states = stack_states(self._stacked_states, state, True)
        self.lives = self.env.ale.lives()

    def run(self):
        print(f"W: {self.id} started.")
        while True:
            self.conn.send(self._stacked_states)
            action = self.conn.recv()
            next_state, r, d, info = self.env.step(action)

            self._stacked_states = stack_states(self._stacked_states, next_state, False)
            self.conn.send((self._stacked_states, np.sign(r), d))
            if d:
                self.reset()
