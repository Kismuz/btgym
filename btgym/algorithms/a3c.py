from __future__ import print_function

from btgym.algorithms import BaseAAC
from btgym.algorithms.losses import aac_loss_def


class A3C(BaseAAC):
    """
    Vanilla Asynchronous Advantage Actor Critic algorithm.

    Based on original code taken from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Paper: https://arxiv.org/abs/1602.01783
    """
    def __init__(self, **kwargs):
        """

        Args:
            env:                    envirionment instance
            task:                   int
            policy_config:          policy estimator class and configuration dictionary
            log:                    parent log
            random_seed:            int or None
            model_gamma:            gamma discount factor
            model_gae_lambda:       GAE lambda
            model_beta:             entropy regularization beta
            opt_max_train_steps:    train steps to run
            opt_decay_steps:        learn ratio decay steps
            opt_end_learn_rate:     final lerarn rate
            opt_learn_rate:         start learn rate
            opt_decay:              optimizer decay, if apll.
            opt_momentum:           optimizer momentum, if apll.
            opt_epsilon:            optimizer epsilon
            rollout_length:         on-policy rollout length
            episode_summary_freq:   int, write episode summary for every i'th episode
            env_render_freq:        int, write environment rendering summary for every i'th train step
            model_summary_freq:     int, write model summary for every i'th train step
            test_mode:              True: Atari, False: BTGym
        """
        super(A3C, self).__init__(
            on_policy_loss=aac_loss_def,
            use_off_policy_aac=False,
            use_reward_prediction=False,
            use_pixel_control=False,
            use_value_replay=False,
            _use_target_policy=False,
            **kwargs
        )