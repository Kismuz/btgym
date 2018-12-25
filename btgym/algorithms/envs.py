# Original code is taken from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent

import numpy as np
import cv2
import gym
from gym import spaces
from btgym import DictSpace, ActionDictSpace


def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    """
    Gym wrapper, pipes Atari into BTgym algorithms, as later expect observations to be DictSpace.
    Makes Atari environment return state as dictionary with single key 'external' holding
    normalized in [0,1] grayscale 42x42 visual output.
    """
    # TODO: INPRoGRESS: dict observation space, include metadata etc.
    def __init__(self, env_id=None):
        """

        Args:
            env_id:     conventional Gym id.
        """
        assert "." not in env_id  # universe environments have dots in names.
        env = gym.make(env_id)
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = DictSpace(
            {'external': spaces.Box(0.0, 1.0, [42, 42, 1], dtype=np.float32)}
        )
        self.asset_names = ['atari_player']
        num_actions = self.action_space.n
        self.action_space = ActionDictSpace(
            base_actions=list(np.arange(num_actions)),
            assets=self.asset_names
        )

    def observation(self, observation):
        return {'external': _process_frame42(observation)}

    def get_initial_action(self):
        return {asset: 0 for asset in self.asset_names}

    def step(self, action):
        # TODO: fix it
        action = action[self.asset_names[0]]
        observation, reward, done, info = self.env.step(action)
        reward = np.asarray(reward)
        return self.observation(observation), reward, done, info
