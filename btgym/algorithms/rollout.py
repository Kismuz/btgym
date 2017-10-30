# UNREAL implementation borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
# https://miyosuda.github.io/
# https://github.com/miyosuda/unreal
#
# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397


import numpy as np

from btgym.algorithms.math_util import discount

# Info:
ExperienceConfig = ['position', 'state', 'action', 'reward', 'value', 'terminal', 'r', 'context',
                    'last_action_reward', 'pixel_change']


class Rollout(dict):
    """Experience rollout as [nested] dictionary of lists.
    """

    def __init__(self):
        super(Rollout, self).__init__()

    def add(self, values_dict, _dict=None):
        """
        Adds single experience to rollout by appending values to dictionary of lists.

        Args:
            values_dict:    [nested] dictionary of values.
        """
        if _dict is None:
            _dict = self
        for key, value in values_dict.items():
            if type(value) == dict:
                if key not in _dict.keys():
                    _dict[key] = {}
                self.add(value, _dict=_dict[key])

            else:
                if key in _dict.keys():
                    _dict[key] += [value]

                else:
                    _dict[key] = [value]

    def add_memory_sample(self, sample):
        """
        Given replay memory sample as list of experience-dictionaries of `length`,
        converts it to rollout of same `length`.
        """
        for frame in sample:
            self.add(frame)

    def process(self, gamma, gae_lambda=1.0):
        """
        Converts rollout to dictionary of ready-to-feed arrays.
        Computes rollout returns and the advantages.

        Returns:
            batch as [nested] dictionary.
        """
        # self._check_it()
        batch = dict()
        for key in self.keys() - {'context', 'reward', 'r', 'value', 'position'}:
            batch[key] = self.as_array(self[key])

        batch['context'] = self['context'][0]  # rollout initial LSTM state

        # Total accumulated empirical return:
        rewards = np.asarray(self['reward'])
        rollout_r = self['r'][-1][0]  # bootstrapped V_next or 0 if terminal
        vpred_t = np.asarray(self['value'] + [rollout_r])
        rewards_plus_v = np.asarray(self['reward'] + [rollout_r])
        batch['r'] = discount(rewards_plus_v, gamma)[:-1]

        # This formula for the advantage is (16) from "Generalized Advantage Estimation" paper:
        # https://arxiv.org/abs/1506.02438
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        batch['advantage'] = discount(delta_t, gamma * gae_lambda)

        return batch

    def extract(self, idx, _struct=None):
        """
        Extracts single experience from rollout.

        Args:
            idx:    experience position

        Returns:
            [nested] dictionary
        """
        # No idx range checks here!
        if _struct is None:
            _struct = self

        if type(_struct) == dict or type(_struct) == type(self):
            frame = {}
            for key, value in _struct.items():
                frame[key] = self.extract(idx, _struct=value)
            return frame

        else:
            return _struct[idx]

    def as_array(self, struct):
        if type(struct) == dict:
            out = {}
            for key, value in struct.items():
                out[key] = self.as_array(value)
            return out

        else:
            return np.asarray(struct)

    def _check_it(self, _struct=None):
        if _struct is None:
            _struct = self
        if type(_struct) == dict or type(_struct) == type(self):
            for key, value in _struct.items():
                print(key, ':')
                self._check_it(_struct=value)
        else:
            try:
                print('length {}, type {}, shape {}\n'.format(len(_struct), type(_struct[0]), _struct[0].shape))
            except:
                print('length {}, type {}\n'.format(len(_struct), type(_struct[0])))
