# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397


import numpy as np

from btgym.algorithms.math_util import discount
from btgym.algorithms.util import batch_pad

# Info:
ExperienceConfig = ['position', 'state', 'action', 'reward', 'value', 'terminal', 'r', 'context',
                    'last_action_reward', 'pixel_change']


def make_rollout_getter(queue):
    """
    Rollout getter constructor.

    Args:
        queue:     instance of `Queue` class to get rollouts from.

    Returns:
        callable, returning instance of Rollout.

    """
    def pull_rollout_from_queue():
        return queue.get(timeout=600.0)

    return pull_rollout_from_queue


class Rollout(dict):
    """
    Experience rollout as [nested] dictionary of lists.
    """

    def __init__(self):
        super(Rollout, self).__init__()
        self.size = 0

    def add(self, values_dict, _dict=None):
        """
        Adds single experience to rollout by appending values to dictionary of lists.

        Args:
            values_dict:    [nested] dictionary of values.
        """
        if _dict is None:
            # Top level:
            _dict = self
            self.size += 1
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

    def process(self, gamma, gae_lambda=1.0, size=None):
        """
        Converts rollout of batch_size=1 to dictionary of ready-to-feed arrays.
        Computes rollout returns and the advantages.
        Pads with zeroes to desired length, if size arg is given.

        Returns:
            batch as [nested] dictionary of np.arrays [, LSTMStateTuples].
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

        # Brush it out:
        batch['time_steps'] = batch['advantage'].shape[0]  # real non-padded time length
        batch['batch_size'] = 1  # rollout is a trajectory

        if size is not None and batch['advantage'].shape[0] != size:
            batch = batch_pad(batch, to_size=size)

        return batch

    def process_rp(self, reward_threshold=0.1):
        """
        Processes rollout process()-alike and estimates reward prediction target for first n-1 frames.

        Args:
            reward_threshold:   reward values such as |r|> reward_threshold are classified as neg. or pos.

        Returns:
            Processed batch with size reduced by one and
            with extra `rp_target` key holding one hot encodings {zero, positive, negative}.
        """

        # Remove last frame:
        last_frame = self.pop_frame(-1)

        batch = self.process(gamma=1)

        # Make one hot vector for target rewards (i.e. reward taken from last of sampled frames):
        r = last_frame['reward']
        rp_t = np.zeros(3)
        if r > reward_threshold:
            rp_t[1] = 1.0  # positive [010]

        elif r < - reward_threshold:
            rp_t[2] = 1.0  # negative [001]

        else:
            rp_t[0] = 1.0  # zero [100]

        batch['rp_target'] = rp_t[None,...]
        batch['time_steps'] = batch['advantage'].shape[0]  # e.g -1 of original

        return batch

    def get_frame(self, idx, _struct=None):
        """
        Extracts single experience from rollout.

        Args:
            idx:    experience position

        Returns:
            frame as [nested] dictionary
        """
        # No idx range checks here!
        if _struct is None:
            _struct = self

        if type(_struct) == dict or type(_struct) == type(self):
            frame = {}
            for key, value in _struct.items():
                frame[key] = self.get_frame(idx, _struct=value)
            return frame

        else:
            return _struct[idx]

    def pop_frame(self, idx, _struct=None):
        """
        Pops single experience from rollout.

        Args:
            idx:    experience position

        Returns:
            frame as [nested] dictionary
        """
        # No idx range checks here!
        if _struct is None:
            _struct = self

        if type(_struct) == dict or type(_struct) == type(self):
            frame = {}
            for key, value in _struct.items():
                frame[key] = self.pop_frame(idx, _struct=value)
            return frame

        else:
            return _struct.pop(idx)

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
