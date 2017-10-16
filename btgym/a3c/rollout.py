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

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features", "pc", "last_ar"])


class PartialRollout(object):
    """
    Experience rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self):
        self.position = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = []
        self.terminal = []
        self.features = []
        self.pixel_change = []
        self.last_actions_rewards = []

    def add(self,
            position,
            state,
            action,
            reward,
            value,
            value_next,
            terminal,
            features,
            pixel_change,
            last_action_reward):
        self.position += [position]
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.r += [value_next]
        self.terminal += [terminal]
        self.features += [features]
        self.pixel_change += [pixel_change]
        self.last_actions_rewards += [last_action_reward]

    def add_memory_sample(self, sample):
        """
        Given replay memory sample as list of frames of `length`,
        converts it to rollout of same `length`.
        """
        for frame in sample:
            self.add(
                frame.position,
                frame.state,
                frame.action,
                frame.reward,
                frame.value,
                frame.r,
                frame.terminal,
                frame.features,
                frame.pixel_change,
                frame.last_action_reward
            )

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def process(self, gamma, gae_lambda=1.0):
        """
        Computes rollout returns and the advantages.
        """
        batch_si = np.asarray(self.states)
        batch_a = np.asarray(self.actions)
        rewards = np.asarray(self.rewards)
        batch_ar = np.asarray(self.last_actions_rewards)  # concatenated 'last action and reward' 's
        pix_change = np.asarray(self.pixel_change)
        rollout_R = self.r[-1][0]  # R, bootstrapped V or 0 if terminal
        vpred_t = np.asarray(self.values + [rollout_R])
        try:
            rewards_plus_v = np.asarray(self.rewards + [rollout_R])
        except:
            print('rollout.r: ', type(self.r))
            print('rollout.rewards[-1]: ', self.rewards[-1], type(self.rewards[-1]))
            print('rollout_R:', rollout_R, rollout_R.shape, type(rollout_R))
            raise RuntimeError('!!!')
        batch_r = self.discount(rewards_plus_v, gamma)[:-1]  # total accumulated empirical returns
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        # this formula for the advantage is (16) from "Generalized Advantage Estimation" paper:
        # https://arxiv.org/abs/1506.02438
        batch_adv = self.discount(delta_t, gamma * gae_lambda)
        features = self.features[0]  # only first LSTM state is needed, others are for replay memory

        return Batch(batch_si, batch_a, batch_adv, batch_r, self.terminal, features, pix_change, batch_ar)

    Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features", "pc", "last_ar"])

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
