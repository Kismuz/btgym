# Asynchronous implementation of Proximal Policy Optimization algorithm.
# paper:
# https://arxiv.org/pdf/1707.06347.pdf
#
# Based on PPO-SGD code from OpenAI `Baselines` repository under MIT licence:
# https://github.com/openai/baselines
#
# Async. framework code comes from OpenAI repository under MIT licence:
# https://github.com/openai/universe-starter-agent
#


from __future__ import print_function

from btgym.algorithms import BaseAAC
from btgym.algorithms.losses import ppo_loss_def



class PPO(BaseAAC):
    """
    Asynchronous implementation of Proximal Policy Optimization algorithm (L^Clip objective)
    augmented with auxiliary control tasks.

    paper:
    https://arxiv.org/pdf/1707.06347.pdf

    Based on PPO-SGD code from OpenAI `Baselines` repository under MIT licence:
    https://github.com/openai/baselines

    Async. framework code comes from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent
    """
    def __init__(self, **kwargs):
        """

        Args:
            env:                    envirionment instance.
            task:                   int
            policy_config:          policy estimator class and configuration dictionary
            log:                    parent log
            vr_loss:                callable returning tensor holding value replay loss and summaries
            rp_loss:                callable returning tensor holding reward prediction loss and summaries
            pc_loss:                callable returning tensor holding pixel_control loss and summaries
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
            replay_memory_size:     in number of experiences
            replay_rollout_length:  off-policy rollout length
            use_off_policy_aac:     use full PPO off-policy training instead of Value-replay
            use_reward_prediction:  use aux. off-policy reward prediction task
            use_pixel_control:      use aux. off-policy pixel control task
            use_value_replay:       use aux. off-policy value replay task (not used, if use_off_policy_aac=True)
            rp_lambda:              reward prediction loss weight
            pc_lambda:              pixel control loss weight
            vr_lambda:              value replay loss weight
            off_aac_lambda:         off-policy PPO loss weight
            gamma_pc:               NOT USED
            rp_reward_threshold:    reward prediction task classification threshold, above which reward is 'non-zero'
            rp_sequence_size:       reward prediction sample size, in number of experiences
            clip_epsilon:           PPO: surrogate L^clip epsilon
            num_epochs:             num. of SGD runs for every train step
            pi_prime_update_period:   int, PPO: pi to pi_old update period in number of train steps
        """
        super(PPO, self).__init__(
            on_policy_loss=ppo_loss_def,
            off_policy_loss=ppo_loss_def,
            _use_target_policy=True,
            **kwargs
        )

