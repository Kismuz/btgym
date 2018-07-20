# Original A3C code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#
# Papers:
# https://arxiv.org/abs/1602.01783
# https://arxiv.org/abs/1611.05397


import numpy as np

from tensorflow.contrib.rnn import LSTMStateTuple
from btgym.algorithms.math_utils import discount
from btgym.algorithms.utils import batch_pad


# Info:
ExperienceConfig = ['position', 'state', 'action', 'reward', 'value', 'terminal', 'r', 'context',
                    'last_action_reward', 'pixel_change']


def make_data_getter(queue):
    """
    Data stream getter constructor.

    Args:
        queue:     instance of `Queue` class to get rollouts from.

    Returns:
        callable, returning dictionary of data.

    """
    def pull_rollout_from_queue(**kwargs):
        return queue.get(timeout=600.0)

    return pull_rollout_from_queue


class Rollout(dict):
    """
    Experience rollout as [nested] dictionary of lists of ndarrays, tuples and rnn states.
    """

    def __init__(self):
        super(Rollout, self).__init__()
        self.size = 0

    def add(self, values, _struct=None):
        """
        Adds single experience frame to rollout.

        Args:
            values:    [nested] dictionary of values.
        """
        if _struct is None:
            # Top level:
            _struct = self
            self.size += 1
            top = True

        else:
            top = False

        try:
            if isinstance(values, dict):
                for key, value in values.items():
                    if key not in _struct.keys():
                        _struct[key] = {}
                    _struct[key] = self.add(value, _struct[key])

            elif isinstance(values, tuple):
                if not isinstance(_struct, tuple):
                    _struct = ['empty' for entry in values]
                _struct = tuple([self.add(*pair) for pair in zip(values, _struct)])

            elif isinstance(values, LSTMStateTuple):
                if not isinstance(_struct, LSTMStateTuple):
                    _struct = LSTMStateTuple(0, 0)
                c = self.add(values[0], _struct[0])
                h = self.add(values[1], _struct[1])
                _struct = LSTMStateTuple(c, h)

            else:
                if isinstance(_struct, list):
                    _struct += [values]

                else:
                    _struct = [values]

        except:
            print('values:\n', values)
            print('_struct:\n', _struct)
            raise RuntimeError

        if not top:
            return _struct

    def add_memory_sample(self, sample):
        """
        Given replay memory sample as list of experience-dictionaries of `length`,
        converts it to rollout of same `length`.
        """
        for frame in sample:
            self.add(frame)

    def process(self, gamma, gae_lambda=1.0, size=None, time_flat=False):
        """
        Converts single-trajectory rollout of experiences to dictionary of ready-to-feed arrays.
        Computes rollout returns and the advantages.
        Pads with zeroes to desired length, if size arg is given.

        Args:
            gamma:          discount factor
            gae_lambda:     GAE lambda
            size:           if given and time_flat=False, pads outputs with zeroes along `time' dim. to exact 'size'.
            time_flat:      reduce time dimension to 1 step by stacking all experiences along batch dimension.

        Returns:
            batch as [nested] dictionary of np.arrays, tuples and LSTMStateTuples. of size:

                [1, time_size, depth] or [1, size, depth] if not time_flatten and `size` is not/given, with single
                `context` entry for entire trajectory, i.e. of size [1, context_depth];

                [batch_size, 1, depth], if time_flatten, with batch_size = time_size and `context` entry for
                every experience frame, i.e. of size [batch_size, context_depth].
        """
        # self._check_it()
        batch = dict()
        for key in self.keys() - {'context', 'reward', 'r', 'value', 'position'}:
            batch[key] = self.as_array(self[key])

        if time_flat:
            batch['context'] = self.as_array(self['context'], squeeze_axis=1)  # LSTM state for every frame

        else:
            batch['context'] = self.get_frame(0)['context'] # just get rollout initial LSTM state

        #print('batch_context:')
        #self._check_it(batch['context'])

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

        # Shape it out:
        if time_flat:
            batch['batch_size'] = batch['advantage'].shape[0]  # time length turned batch size
            batch['time_steps'] = np.ones(batch['batch_size'])

        else:
            batch['time_steps'] = batch['advantage'].shape[0]  # real non-padded time length
            batch['batch_size'] = 1  # want rollout as a trajectory

        if size is not None and not time_flat and batch['advantage'].shape[0] != size:
            # Want all batches to be exact size for further batch stacking:
            batch = batch_pad(batch, to_size=size)

        return batch

    def process_rp(self, reward_threshold=0.1):
        """
        Processes rollout process()-alike and estimates reward prediction target for first n-1 frames.

        Args:
            reward_threshold:   reward values such as |r|> reward_threshold are classified as neg. or pos.

        Returns:
            Processed batch with size reduced by one and with extra `rp_target` key
            holding one hot encodings for classes {zero, positive, negative}.
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

        if isinstance(_struct, dict) or type(_struct) == type(self):
            frame = {}
            for key, value in _struct.items():
                frame[key] = self.get_frame(idx, value)
            return frame

        elif isinstance(_struct, tuple):
            return tuple([self.get_frame(idx, value) for value in _struct])

        elif isinstance(_struct, LSTMStateTuple):
            return LSTMStateTuple(self.get_frame(idx, _struct[0]), self.get_frame(idx, _struct[1]))

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

        if isinstance(_struct, dict) or type(_struct) == type(self):
            frame = {}
            for key, value in _struct.items():
                frame[key] = self.pop_frame(idx, value)
            return frame

        elif isinstance(_struct, tuple):
            return tuple([self.pop_frame(idx, value) for value in _struct])

        elif isinstance(_struct, LSTMStateTuple):
            return LSTMStateTuple(self.pop_frame(idx, _struct[0]), self.pop_frame(idx, _struct[1]))

        else:
            return _struct.pop(idx)

    def as_array(self, struct, squeeze_axis=None):
        if isinstance(struct, dict):
            out = {}
            for key, value in struct.items():
                out[key] = self.as_array(value, squeeze_axis)
            return out

        elif isinstance(struct, tuple):
            return tuple([self.as_array(value, squeeze_axis) for value in struct])

        elif isinstance(struct, LSTMStateTuple):
            return LSTMStateTuple(self.as_array(struct[0], squeeze_axis), self.as_array(struct[1], squeeze_axis))

        else:
            if squeeze_axis is not None:
                return np.squeeze(np.asarray(struct), axis=squeeze_axis)

            else:
                return np.asarray(struct)

    def _check_it(self, _struct=None):
        if _struct is None:
            _struct = self
        if type(_struct) == dict or type(_struct) == type(self):
            for key, value in _struct.items():
                print(key, ':')
                self._check_it(_struct=value)

        elif type(_struct) == tuple or type(_struct) == list:
            print('tuple/list:')
            for value in _struct:
                self._check_it(_struct=value)

        else:
            try:
                print('length: {}, type: {}, shape of element: {}\n'.format(len(_struct), type(_struct[0]), _struct[0].shape))
            except:
                print('length: {}, type: {}\n'.format(len(_struct), type(_struct[0])))
