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
    """
    Experience rollout as dictionary of lists of values.
    We run our agent, and process its experience once it has processed enough steps.
    """
    def __init__(self, experience_config=ExperienceConfig):
        """
        Args:
            experience_config:  list of experience fields to store.
        """
        super(Rollout, self).__init__()
        #for key in experience_config:
        #    self[key] = []

    def add(self, values_dict):
        """
        Adds single experience to rollout.
        Args:
            values_dict:    dictionary of values.
        """
        for key in values_dict.keys():
            try:
                self[key] += [values_dict[key]]
            except:
                self[key] = [values_dict[key]]

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
        Returns: dictionary of batched data.
        """
        #self._check_it()
        batch = dict()
        for key in self.keys() - {'context', 'reward', 'r', 'value', 'position'}:
            batch[key] = np.asarray(self[key])

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

    def _check_it(self):
        for key, list in self.items():
            try:
                print('{}: length {}, type {}, shape {}\n'.format(key, len(list), type(list[0]), list[0].shape))
            except:
                print('{}: length {}, type {}\n'.format(key, len(list), type(list[0])))

