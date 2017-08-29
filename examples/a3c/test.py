import gym
from gym import error, spaces
import numpy as np


class test_env(gym.Env):
    def __init__(self):
        # self.metadata = {}
        self.observation_space = spaces.Box(shape=(100, 100, 3), low=0, high=10, )
        self.action_space = spaces.Discrete(6)
        self.ep_step = 0

    def _reset(self):
        self.ep_step = 0
        return np.zeros(self.observation_space.shape)

    def _step(self, action):
        if self.ep_step >= 10:
            done = True

        else:
            done = False

        o = np.ones(self.observation_space.shape) * self.ep_step / 10
        r = self.ep_step / 100
        self.ep_step += 1

        return (o, r, done, '==test_env==')

