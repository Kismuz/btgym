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

import scipy.signal
import numpy as np
from collections import namedtuple

ExperienceConfig = ['position', 'state', 'action', 'reward', 'value', 'terminal', 'r', 'context',
                    'pixel_change', 'last_action_reward']

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features", "pc", "last_ar"])


class Rollout(dict):
    """
    Experience rollout as dictionary of lists of values.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, experience_config=ExperienceConfig):
        """
        Args:
            experience_config:  list of experience fields to store.
        """
        super(Rollout, self).__init__()
        for key in experience_config:
            self[key] = []

    def add(self, values_dict):
        """
        Adds single experience to rollout.
        Args:
            values_dict:    dictionary of values, keys must be consistent with self.experience_config.
        """
        for key in self.keys():
            self[key] += [values_dict[key]]

    def add_memory_sample(self, sample):
        """
        Given replay memory sample as list of experience-dictionaries of `length`,
        converts it to rollout of same `length`.
        """
        for frame in sample:
            self.add(frame)

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def process(self, gamma, gae_lambda=1.0):
        """
        Computes rollout returns and the advantages.
        Returns: named tuple.
        """
        batch_si = np.asarray(self['state'])
        batch_a = np.asarray(self['action'])
        rewards = np.asarray(self['reward'])
        batch_ar = np.asarray(self['last_action_reward'])  # concatenated 'last action and reward' 's
        pix_change = np.asarray(self['pixel_change'])
        rollout_r = self['r'][-1][0]  # bootstrapped V_next or 0 if terminal
        vpred_t = np.asarray(self['value'] + [rollout_r])
        #try:
        rewards_plus_v = np.asarray(self['reward'] + [rollout_r])
        #except:
        #    print('rollout.r: ', type(self.r))
        #    print('rollout.rewards[-1]: ', self.rewards[-1], type(self.rewards[-1]))
        #    print('rollout_R:', rollout_r, rollout_r.shape, type(rollout_r))
        #    raise RuntimeError('!!!')
        batch_r = self.discount(rewards_plus_v, gamma)[:-1]  # total accumulated empirical returns
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage is (16) from "Generalized Advantage Estimation" paper:
        # https://arxiv.org/abs/1506.02438
        batch_adv = self.discount(delta_t, gamma * gae_lambda)
        init_context = self['context'][0]  # rollout initial LSTM state

        return Batch(batch_si, batch_a, batch_adv, batch_r, self['terminal'], init_context, pix_change, batch_ar)

    """
    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r  # !!
        self.state_next = other.state_next
        self.terminal = other.terminal
        self.features = other.features
        self.pixel_change.extend(other.pixel_change)
        self.last_actions_rewards = other.last_action_reward
    """
