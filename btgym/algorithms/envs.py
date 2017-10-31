# Original code is taken from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent

import numpy as np
import cv2
import gym
from gym import spaces


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
    def __init__(self, env_id=None):
        """
        Gym wrapper class.
        Makes Atari environment normalized grayscale 42x42 in [0,1].

        Args:
            env_id:     conventional Gym id.
        """
        assert "." not in env_id  # universe environments have dots in names.
        env = gym.make(env_id)
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation):
        return _process_frame42(observation)