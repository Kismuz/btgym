import gym
from gym import error, spaces
import numpy as np


class test_env(gym.Env):
    """
    Simple atari-like tester environment for checking a3c output consistency.
    Use with a3c launcher configurator:

        cluster_config = dict(
        host='127.0.0.1',
        port=42222,
        num_workers=8,
        num_ps=1,
        log_dir='./tmp/a3c_testing_',
        )

        env_config = dict(gym_id='test-v01')
        launcher = Launcher(
            cluster_config=cluster_config,
            env_config=env_config,
            train_steps=500,
            opt_learn_rate=1e-4,
            rollout_length=20,
            test_mode=True,
            model_summary_freq=50,
            episode_summary_freq=2,
            env_render_freq=10,
            verbose=1
        )

    """
    def __init__(self):
        # self.metadata = {}
        self.observation_space = spaces.Box(shape=(100, 100, 3), low=0, high=10, )
        self.action_space = spaces.Discrete(6)
        self.ep_step = 0

    def _reset(self):
        self.ep_step = 0
        return np.zeros(self.observation_space.shape)

    def _step(self, action):
        if self.ep_step >= 10: # max episode length: 10
            done = True

        else:
            done = False

        o = np.ones(self.observation_space.shape) * self.ep_step / 10
        r = self.ep_step / 100
        self.ep_step += 1

        return (o, r, done, '==test_env==')

