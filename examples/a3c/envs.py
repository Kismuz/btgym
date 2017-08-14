# Original code is taken from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent

import cv2
from gym.spaces.box import Box
import numpy as np
import gym

from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger


def create_env(env_id,  **kwargs):
    """
    Sstate preprocessor wrapper.
    Only Atari environments are supported.
    """
    assert "." not in env_id  # universe environments have dots in names.
    return create_atari_env(env_id)

def create_atari_env(env_id):

    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = Unvectorize(env)
    return env

def _process_frame42(frame):
    # TODO: get rid of cv2 and universe wrappers
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

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]
