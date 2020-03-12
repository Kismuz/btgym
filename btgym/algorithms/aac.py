###############################################################################
#
# Copyright (C) 2017 Andrew Muzikin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from logbook import Logger, StreamHandler

from btgym.algorithms.memory import Memory
from btgym.algorithms.rollout import make_data_getter
from btgym.algorithms.runner import BaseEnvRunnerFn, RunnerThread
from btgym.algorithms.math_utils import log_uniform
from btgym.algorithms.nn.losses import value_fn_loss_def, rp_loss_def, pc_loss_def, aac_loss_def, ppo_loss_def
from btgym.algorithms.utils import feed_dict_rnn_context, feed_dict_from_nested, batch_stack
from btgym.spaces import DictSpace as BaseObSpace
from btgym.spaces import ActionDictSpace as BaseAcSpace


class BaseAAC(object):
    """
    Base Asynchronous Advantage Actor Critic algorithm framework class with auxiliary control tasks and
    option to run several instances of environment for every worker in vectorized fashion, PAAC-like.
    Can be configured to run with different losses and policies.

    Auxiliary tasks implementation borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
    https://miyosuda.github.io/
    https://github.com/miyosuda/unreal

    Original A3C code comes from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Papers:
    https://arxiv.org/abs/1602.01783
    https://arxiv.org/abs/1611.05397
    """
    def __init__(self,
                 env,
                 task,
                 policy_config,
                 log_level,
                 name='AAC',
                 on_policy_loss=aac_loss_def,
                 off_policy_loss=aac_loss_def,
                 vr_loss=value_fn_loss_def,
                 rp_loss=rp_loss_def,
                 pc_loss=pc_loss_def,
                 runner_config=None,
                 runner_fn_ref=BaseEnvRunnerFn,
                 cluster_spec=None,
                 random_seed=None,
                 model_gamma=0.99,  # decay
                 model_gae_lambda=1.00,  # GAE lambda
                 model_beta=0.01,  # entropy regularizer
                 opt_max_env_steps=10 ** 7,
                 opt_decay_steps=None,
                 opt_end_learn_rate=None,
                 opt_learn_rate=1e-4,
                 opt_decay=0.99,
                 opt_momentum=0.0,
                 opt_epsilon=1e-8,
                 rollout_length=20,
                 time_flat=False,
                 episode_train_test_cycle=(1,0),
                 episode_summary_freq=2,  # every i`th environment episode
                 env_render_freq=10,  # every i`th environment episode
                 model_summary_freq=100,  # every i`th algorithm iteration
                 test_mode=False,  # gym_atari test mode
                 replay_memory_size=2000,
                 replay_batch_size=None,
                 replay_rollout_length=None,
                 use_off_policy_aac=False,
                 use_reward_prediction=False,
                 use_pixel_control=False,
                 use_value_replay=False,
                 rp_lambda=1.0,  # aux tasks loss weights
                 pc_lambda=1.0,
                 vr_lambda=1.0,
                 off_aac_lambda=1,
                 gamma_pc=0.9,  # pixel change gamma-decay - not used
                 rp_reward_threshold=0.1,  # r.prediction: abs.rewards values bigger than this are considered non-zero
                 rp_sequence_size=3,  # r.prediction sampling
                 clip_epsilon=0.1,
                 num_epochs=1,
                 pi_prime_update_period=1,
                 global_step_op=None,
                 global_episode_op=None,
                 inc_episode_op=None,
                 _use_global_network=True,
                 _use_target_policy=False,  # target policy tracking behavioral one with delay
                 _use_local_memory=False,  # in-place memory
                 aux_render_modes=None,
                 **kwargs):
        """

        Args:
            env:                    environment instance or list of instances
            task:                   int, parent worker id
            policy_config:          policy estimator class and configuration dictionary
            log_level:              int, logbook.level
            name:                   str, class-wide name-scope
            on_policy_loss:         callable returning tensor holding on_policy training loss graph and summaries
            off_policy_loss:        callable returning tensor holding off_policy training loss graph and summaries
            vr_loss:                callable returning tensor holding value replay loss graph and summaries
            rp_loss:                callable returning tensor holding reward prediction loss graph and summaries
            pc_loss:                callable returning tensor holding pixel_control loss graph and summaries
            runner_config:          runner class and configuration dictionary,
            runner_fn_ref:          callable defining environment runner execution logic,
                                    valid only if no 'runner_config' arg is provided
            cluster_spec:           dict, full training cluster spec (may be used by meta-trainer)
            random_seed:            int or None
            model_gamma:            scalar, gamma discount factor
            model_gae_lambda:       scalar, GAE lambda
            model_beta:             entropy regularization beta, scalar or [high_bound, low_bound] for log_uniform.
            opt_max_env_steps:      int, total number of environment steps to run training on.
            opt_decay_steps:        int, learn ratio decay steps, in number of environment steps.
            opt_end_learn_rate:     scalar, final learn rate
            opt_learn_rate:         start learn rate, scalar or [high_bound, low_bound] for log_uniform distr.
            opt_decay:              scalar, optimizer decay, if apll.
            opt_momentum:           scalar, optimizer momentum, if apll.
            opt_epsilon:            scalar, optimizer epsilon
            rollout_length:         int, on-policy rollout length
            time_flat:              bool, flatten rnn time-steps in rollouts of size 1 - see `Notes` below
            episode_train_test_cycle:   tuple or list as (train_number, test_number), def=(1,0): enables infinite
                                        loop such as: run `train_number` of train data episodes,
                                        than `test_number` of test data episodes, repeat. Should be consistent
                                        with provided dataset parameters (test data should exist if `test_number > 0`)
            episode_summary_freq:   int, write episode summary for every i'th episode
            env_render_freq:        int, write environment rendering summary for every i'th train step
            model_summary_freq:     int, write model summary for every i'th train step
            test_mode:              bool, True: Atari, False: BTGym
            replay_memory_size:     int, in number of experiences
            replay_batch_size:      int, mini-batch size for off-policy training, def = 1
            replay_rollout_length:  int off-policy rollout length by def. equals on_policy_rollout_length
            use_off_policy_aac:     bool, use full AAC off-policy loss instead of Value-replay
            use_reward_prediction:  bool, use aux. off-policy reward prediction task
            use_pixel_control:      bool, use aux. off-policy pixel control task
            use_value_replay:       bool, use aux. off-policy value replay task (not used if use_off_policy_aac=True)
            rp_lambda:              reward prediction loss weight, scalar or [high, low] for log_uniform distr.
            pc_lambda:              pixel control loss weight, scalar or [high, low] for log_uniform distr.
            vr_lambda:              value replay loss weight, scalar or [high, low] for log_uniform distr.
            off_aac_lambda:         off-policy AAC loss weight, scalar or [high, low] for log_uniform distr.
            gamma_pc:               NOT USED
            rp_reward_threshold:    scalar, reward prediction classification threshold, above which reward is 'non-zero'
            rp_sequence_size:       int, reward prediction sample size, in number of experiences
            clip_epsilon:           scalar, PPO: surrogate L^clip epsilon
            num_epochs:             int, num. of SGD runs for every train step, val. > 1 should be used with caution.
            pi_prime_update_period: int, PPO: pi to pi_old update period in number of train steps, def: 1
            global_step_op:         external tf.variable holding global step counter
            global_episode_op:      external tf.variable holding global episode counter
            inc_episode_op:         external tf.op incrementing global step counter
            _use_global_network:    bool, either to use parameter server policy instance
            _use_target_policy:     bool, PPO: use target policy (aka pi_old), delayed by `pi_prime_update_period` delay
            _use_local_memory:      bool: use in-process replay memory instead of runner-based one
            aux_render_modes:      additional visualisations to include in per-episode rendering summary

        Note:
            - On `time_flat` arg:

                Note that previous explanation of this arg was erroneous;
                Time_flat=False:
                    Implements Truncated BPTT with backpropagation depth equal to rollout length.
                    In this case we need to feed initial rnn_states for rollouts only.
                    Thus, when time_flat=False, we unroll RNN in specified number of time-steps for every rollout.

                Time_flat=True:
                Basicaly forces TBPTT with rollout depth = 1.
                Not recommended to use as it prevents policy from learning long-range dependencies.
        """
        # Logging:
        self.log_level = log_level
        self.name = name
        self.task = task
        self.cluster_spec = cluster_spec
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('{}_{}'.format(self.name, self.task), level=self.log_level)

        # Get direct traceback:
        try:
            self.random_seed = random_seed
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
                tf.set_random_seed(self.random_seed)
            self.log.debug('rnd_seed:{}, log_u_sample_(0,1]x5: {}'.
                           format(random_seed, log_uniform([1e-10,1], 5)))

            if kwargs != {}:
                self.log.warning('Unexpected kwargs found: {}, ignored.'.format(kwargs))

            self.env_list = env
            try:
                assert isinstance(self.env_list, list)

            except AssertionError:
                self.env_list = [env]

            self.ref_env = self.env_list[0]  # reference instance to get obs shapes etc.

            try:
                assert isinstance(self.ref_env.observation_space, BaseObSpace)

            except AssertionError:
                self.log.exception(
                    'expected environment observation space of type {}, got: {}'.\
                    format(BaseObSpace, type(self.ref_env.observation_space))
                )
                raise AssertionError

            try:
                assert isinstance(self.ref_env.action_space, BaseAcSpace)

            except AssertionError:
                self.log.exception(
                    'expected environment observation space of type {}, got: {}'.\
                    format(BaseAcSpace, type(self.ref_env.action_space))
                )
                raise AssertionError

            self.policy_class = policy_config['class_ref']
            self.policy_kwargs = policy_config['kwargs']

            # Losses:
            self.on_policy_loss = on_policy_loss
            self.off_policy_loss = off_policy_loss
            self.vr_loss = vr_loss
            self.rp_loss = rp_loss
            self.pc_loss = pc_loss

            if runner_config is None:
                # Runner will be async. ThreadRunner class with runner_fn logic:
                self.runner_config = {
                    'class_ref': RunnerThread,
                    'kwargs': {
                        'runner_fn_ref': runner_fn_ref,
                    }
                }
            else:
                self.runner_config = runner_config

            # AAC specific:
            self.model_gamma = model_gamma  # decay
            self.model_gae_lambda = model_gae_lambda  # general advantage estimator lambda
            self.model_beta = log_uniform(model_beta, 1)  # entropy reg.

            self.time_flat = time_flat

            # Optimizer
            self.opt_max_env_steps = opt_max_env_steps
            self.opt_learn_rate = log_uniform(opt_learn_rate, 1)

            if opt_end_learn_rate is None:
                self.opt_end_learn_rate = self.opt_learn_rate
            else:
                self.opt_end_learn_rate = opt_end_learn_rate

            if opt_decay_steps is None:
                self.opt_decay_steps = self.opt_max_env_steps
            else:
                self.opt_decay_steps = opt_decay_steps

            self.opt_decay = opt_decay
            self.opt_epsilon = opt_epsilon
            self.opt_momentum = opt_momentum
            self.rollout_length = rollout_length

            # Data sampling control:
            self.num_train_episodes = episode_train_test_cycle[0]
            self.num_test_episodes = episode_train_test_cycle[-1]
            try:
                assert self.num_train_episodes + self.num_test_episodes > 0 and \
                    self.num_train_episodes >= 0 and \
                    self.num_test_episodes >= 0

            except AssertionError:
                self.log.exception(
                    'Train/test episode cycle values could not be both zeroes or negative, got: train={}, test={}'.\
                    format(self.num_train_episodes, self.num_test_episodes)
                )
                raise AssertionError

            self.current_train_episode = 0
            self.current_test_episode = 0

            # Summaries :
            self.episode_summary_freq = episode_summary_freq
            self.env_render_freq = env_render_freq
            self.model_summary_freq = model_summary_freq

            # If True - use ATARI gym env.:
            self.test_mode = test_mode

            # UNREAL/AUX and Off-policy specific:
            self.off_aac_lambda = log_uniform(off_aac_lambda, 1)
            self.rp_lambda = log_uniform(rp_lambda, 1)
            self.pc_lambda = log_uniform(pc_lambda, 1)
            self.vr_lambda = log_uniform(vr_lambda, 1)
            self.gamma_pc = gamma_pc
            self.replay_memory_size = replay_memory_size

            if replay_rollout_length is not None:
                self.replay_rollout_length = replay_rollout_length

            else:
                self.replay_rollout_length = rollout_length # by default off-rollout equals on-policy one

            self.rp_sequence_size = rp_sequence_size
            self.rp_reward_threshold = rp_reward_threshold

            if replay_batch_size is not None:
                self.replay_batch_size = replay_batch_size

            else:
                self.replay_batch_size = len(self.env_list)  # by default off-batch equals on-policy one

            # PPO related:
            self.clip_epsilon = clip_epsilon
            self.num_epochs = num_epochs
            self.pi_prime_update_period = pi_prime_update_period

            # On/off switchers for off-policy training and auxiliary tasks:
            self.use_off_policy_aac = use_off_policy_aac
            self.use_reward_prediction = use_reward_prediction
            self.use_pixel_control = use_pixel_control
            if use_off_policy_aac:
                self.use_value_replay = False  # v-replay is redundant in this case
            else:
                self.use_value_replay = use_value_replay

            self.use_any_aux_tasks = use_value_replay or use_pixel_control or use_reward_prediction
            self.use_local_memory = _use_local_memory
            self.use_memory = (self.use_any_aux_tasks or self.use_off_policy_aac) and not self.use_local_memory

            self.use_target_policy = _use_target_policy
            self.use_global_network = _use_global_network

            self.log.notice('learn_rate: {:1.6f}, entropy_beta: {:1.6f}'.format(self.opt_learn_rate, self.model_beta))

            if self.use_off_policy_aac:
                self.log.notice('off_aac_lambda: {:1.6f}'.format(self.off_aac_lambda,))

            if self.use_any_aux_tasks:
                self.log.notice('vr_lambda: {:1.6f}, pc_lambda: {:1.6f}, rp_lambda: {:1.6f}'.
                              format(self.vr_lambda, self.pc_lambda, self.rp_lambda))

            if aux_render_modes is not None:
                self.aux_render_modes = list(aux_render_modes)
            else:
                self.aux_render_modes = []

            #self.log.notice(
            #    'AAC_{}: max_steps: {}, decay_steps: {}, end_rate: {:1.6f},'.
            #        format(self.task, self.opt_max_env_steps, self.opt_decay_steps, self.opt_end_learn_rate))

            self.worker_device = "/job:worker/task:{}/cpu:0".format(task)

            # Update policy configuration
            self.policy_kwargs.update(
                {
                    'ob_space': self.ref_env.observation_space,
                    'ac_space': self.ref_env.action_space,
                    'rp_sequence_size': self.rp_sequence_size,
                    'aux_estimate': self.use_any_aux_tasks,
                    'static_rnn': self.time_flat,
                    'task': self.task,
                    'cluster_spec': self.cluster_spec
                }
            )

            if global_step_op is not None:
                self.global_step = global_step_op

            if global_episode_op is not None:
                self.global_episode = global_episode_op

            if inc_episode_op is not None:
                self.inc_episode = inc_episode_op

            # Should be defined later:
            self.sync = None
            self.sync_pi = None
            self.sync_pi_prime = None
            self.grads = None
            self.summary_writer = None
            self.local_steps = 0

            # Start building graphs:
            self.log.debug('started building graphs...')
            if self.use_global_network:
                # PS:
                with tf.device(tf.train.replica_device_setter(1, worker_device=self.worker_device)):
                    self.network = pi_global = self._make_policy('global')
                    if self.use_target_policy:
                        self.network_prime = self._make_policy('global_prime')
                    else:
                        self.network_prime = self._make_dummy_policy()
            else:
                self.network = pi_global = self._make_dummy_policy()
                self.network_prime = self._make_dummy_policy()

            # Worker:
            with tf.device(self.worker_device):
                with tf.variable_scope(self.name):
                    self.local_network = pi = self._make_policy('local')

                    if self.use_target_policy:
                        self.local_network_prime = pi_prime = self._make_policy('local_prime')

                    else:
                        self.local_network_prime = pi_prime = self._make_dummy_policy()

                    self.worker_device_callback_0()  # if need more networks etc.

                    # Meant for Batch-norm layers:
                    pi.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*local.*')

                    # Just in case:
                    self.dummy_pi = self._make_dummy_policy()

                    self.log.debug('local_network_upd_ops_collection:\n{}'.format(pi.update_ops))
                    self.log.debug('\nlocal_network_var_list_to_save:')
                    for v in pi.var_list:
                        self.log.debug('{}: {}'.format(v.name, v.get_shape()))

                    #  Learning rate annealing:
                    self.learn_rate_decayed = tf.train.polynomial_decay(
                        self.opt_learn_rate,
                        self.global_step + 1,
                        self.opt_decay_steps,
                        self.opt_end_learn_rate,
                        power=1,
                        cycle=False,
                    )
                    # Freeze training if train_phase is False:
                    self.train_learn_rate = self.learn_rate_decayed * tf.cast(pi.train_phase, tf.float64)
                    self.log.debug('learn rate ok')

                    # Define loss and related summaries
                    self.loss, self.loss_summaries = self._make_loss(pi=pi, pi_prime=pi_prime)

                    if self.use_global_network:
                        # Define train, sync ops:
                        self.train_op = self._make_train_op(pi=pi, pi_prime=pi_prime, pi_global=pi_global)

                    else:
                        self.train_op = []

                    # Model stat. summary, episode summary:
                    self.model_summary_op, self.ep_summary = self._combine_summaries(
                        policy=pi,
                        model_summaries=self.loss_summaries
                    )

                    # Make thread-runner processes:
                    self.runners = self._make_runners(policy=pi)

                    # Make rollouts provider[s] for async runners:
                    if self.runner_config['class_ref'] == RunnerThread:
                        # Make rollouts provider[s] for async threaded runners:
                        self.data_getter = [make_data_getter(runner.queue) for runner in self.runners]
                    else:
                        # Else assume runner is in-thread synchro type and  supports .get data() method:
                        self.data_getter = [runner.get_data for runner in self.runners]

                    self.log.debug('trainer.__init__() ok')

        except:
            msg = 'Base class __init__() exception occurred.' +\
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)

    def worker_device_callback_0(self):
        pass

    def _make_loss(self, **kwargs):
        return self._make_base_loss(name=self.name, verbose=True, **kwargs)

    def _make_base_loss(self, pi, pi_prime, name='base', verbose=True):
        """
        Defines base AAC on- and off-policy loss, auxiliary VR, RP and PC losses, placeholders and summaries.

        Args:
            pi:                 policy network obj.
            pi_prime:           optional policy network obj.
            name:               str, name scope
            verbose:            summary level

        Returns:
            tensor holding estimated loss graph
            list of related summaries
        """
        with tf.name_scope(name):
            # On-policy AAC loss definition:
            pi.on_pi_act_target = tf.placeholder(
                tf.float32, [None, self.ref_env.action_space.one_hot_depth], name="on_policy_action_pl"
            )
            pi.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
            pi.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")

            clip_epsilon = tf.cast(self.clip_epsilon * self.learn_rate_decayed / self.opt_learn_rate, tf.float32)

            on_pi_loss, on_pi_summaries = self.on_policy_loss(
                act_target=pi.on_pi_act_target,
                adv_target=pi.on_pi_adv_target,
                r_target=pi.on_pi_r_target,
                pi_logits=pi.on_logits,
                pi_vf=pi.on_vf,
                pi_prime_logits=pi_prime.on_logits,
                entropy_beta=self.model_beta,
                epsilon=clip_epsilon,
                name='on_policy',
                verbose=verbose
            )
            # Start accumulating total loss:
            loss = on_pi_loss
            model_summaries = on_pi_summaries

            # Off-policy losses:
            pi.off_pi_act_target = tf.placeholder(
                tf.float32, [None, self.ref_env.action_space.one_hot_depth], name="off_policy_action_pl")
            pi.off_pi_adv_target = tf.placeholder(tf.float32, [None], name="off_policy_advantage_pl")
            pi.off_pi_r_target = tf.placeholder(tf.float32, [None], name="off_policy_return_pl")

            if self.use_off_policy_aac:
                # Off-policy AAC loss graph mirrors on-policy:
                off_pi_loss, off_pi_summaries = self.off_policy_loss(
                    act_target=pi.off_pi_act_target,
                    adv_target=pi.off_pi_adv_target,
                    r_target=pi.off_pi_r_target,
                    pi_logits=pi.off_logits,
                    pi_vf=pi.off_vf,
                    pi_prime_logits=pi_prime.off_logits,
                    entropy_beta=self.model_beta,
                    epsilon=clip_epsilon,
                    name='off_policy',
                    verbose=False
                )
                loss = loss + self.off_aac_lambda * off_pi_loss
                model_summaries += off_pi_summaries

            if self.use_pixel_control:
                # Pixel control loss:
                pi.pc_action = tf.placeholder(tf.float32, [None, self.ref_env.action_space.tensor_shape[0]], name="pc_action")
                pi.pc_target = tf.placeholder(tf.float32, [None, None, None], name="pc_target")

                pc_loss, pc_summaries = self.pc_loss(
                    actions=pi.pc_action,
                    targets=pi.pc_target,
                    pi_pc_q=pi.pc_q,
                    name='off_policy',
                    verbose=verbose
                )
                loss = loss + self.pc_lambda * pc_loss
                # Add specific summary:
                model_summaries += pc_summaries

            if self.use_value_replay:
                # Value function replay loss:
                pi.vr_target = tf.placeholder(tf.float32, [None], name="vr_target")
                vr_loss, vr_summaries = self.vr_loss(
                    r_target=pi.vr_target,
                    pi_vf=pi.vr_value,
                    name='off_policy',
                    verbose=verbose
                )
                loss = loss + self.vr_lambda * vr_loss
                model_summaries += vr_summaries

            if self.use_reward_prediction:
                # Reward prediction loss:
                pi.rp_target = tf.placeholder(tf.float32, [None, 3], name="rp_target")

                rp_loss, rp_summaries = self.rp_loss(
                    rp_targets=pi.rp_target,
                    pi_rp_logits=pi.rp_logits,
                    name='off_policy',
                    verbose=verbose
                )
                loss = loss + self.rp_lambda * rp_loss
                model_summaries += rp_summaries

        return loss, model_summaries

    def _make_train_op(self, pi, pi_prime, pi_global):
        """
        Defines training op graph and supplementary sync operations.

        Args:
            pi:                 policy network obj.
            pi_prime:           optional policy network obj.
            pi_global:          shared policy network obj. hosted by parameter server

        Returns:
            tensor holding training op graph;
        """

        # Each worker gets a different set of adam optimizer parameters:
        self.optimizer = tf.train.AdamOptimizer(self.train_learn_rate, epsilon=1e-5)

        # self.optimizer = tf.train.RMSPropOptimizer(
        #    learning_rate=train_learn_rate,
        #    decay=self.opt_decay,
        #    momentum=self.opt_momentum,
        #    epsilon=self.opt_epsilon,
        # )

        # Clipped gradients:
        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, pi.var_list),
            40.0
        )
        self.grads_global_norm = tf.global_norm(self.grads)
        # Copy weights from the parameter server to the local model
        self.sync = self.sync_pi = tf.group(
            *[v1.assign(v2) for v1, v2 in zip(pi.var_list, pi_global.var_list)]
        )
        if self.use_target_policy:
            # Copy weights from new policy model to target one:
            self.sync_pi_prime = tf.group(
                *[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)]
            )
        grads_and_vars = list(zip(self.grads, pi_global.var_list))

        # Set global_step increment equal to observation space batch size:
        obs_space_keys = list(pi.on_state_in.keys())

        # Handles case when 'external' is nested or flat dict:
        assert 'external' in obs_space_keys, \
            'Expected observation space to contain `external` mode, got: {}'.format(obs_space_keys)
        if isinstance(pi.on_state_in['external'], dict):
            stream = pi.on_state_in['external'][list(pi.on_state_in['external'].keys())[0]]
        else:
            stream = pi.on_state_in['external']
        self.inc_step = self.global_step.assign_add(tf.shape(stream)[0])

        train_op = self.optimizer.apply_gradients(grads_and_vars)
        self.log.debug('train_op defined')
        return train_op

    def _combine_summaries(self, policy=None, model_summaries=None):
        """
        Defines model-wide and episode-related summaries

        Returns:
            model_summary op
            episode_summary op
        """
        if model_summaries is not None:
            if self.use_global_network:
                # Model-wide statistics:
                with tf.name_scope('model'):
                    model_summaries += [
                        tf.summary.scalar("grad_global_norm", self.grads_global_norm),
                        # TODO: add gradient variance summary
                        #tf.summary.scalar("learn_rate", self.train_learn_rate),
                        tf.summary.scalar("learn_rate", self.learn_rate_decayed),  # cause actual rate is a jaggy due to test freezes
                        tf.summary.scalar("total_loss", self.loss),
                        # tf.summary.scalar('roll_reward', tf.reduce_mean(self.local_network.on_last_reward_in)),
                        # tf.summary.scalar('roll_advantage', tf.reduce_mean(self.local_network.on_pi_adv_target)),
                    ]
                    if policy is not None:
                        model_summaries += [tf.summary.scalar("var_global_norm", tf.global_norm(policy.var_list))]
        else:
            model_summaries = []
        # Model stat. summary:
        model_summary = tf.summary.merge(model_summaries, name='model_summary')

        # Episode-related summaries:
        ep_summary = dict(
            # Summary placeholders
            render_atari=tf.placeholder(tf.uint8, [None, None, None, 1]),
            total_r=tf.placeholder(tf.float32, ),
            cpu_time=tf.placeholder(tf.float32, ),
            final_value=tf.placeholder(tf.float32, ),
            steps=tf.placeholder(tf.int32, ),
        )
        if self.test_mode:
            # For Atari:
            ep_summary['render_op'] = tf.summary.image("model/state", ep_summary['render_atari'])

        else:
            # BTGym rendering:
            ep_summary.update(
                {
                    mode: tf.placeholder(tf.uint8, [None, None, None, None], name=mode + '_pl')
                    for mode in self.env_list[0].render_modes + self.aux_render_modes
                }
            )
            ep_summary['render_op'] = tf.summary.merge(
                [tf.summary.image(mode, ep_summary[mode])
                 for mode in self.env_list[0].render_modes + self.aux_render_modes]
            )
        # Episode stat. summary:
        ep_summary['btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_train/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode_train/cpu_time_sec', ep_summary['cpu_time']),
                tf.summary.scalar('episode_train/final_value', ep_summary['final_value']),
                tf.summary.scalar('episode_train/env_steps', ep_summary['steps'])
            ],
            name='episode_train_btgym'
        )
        # Test episode stat. summary:
        ep_summary['test_btgym_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode_test/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode_test/final_value', ep_summary['final_value']),
                tf.summary.scalar('episode_test/env_steps', ep_summary['steps'])
            ],
            name='episode_test_btgym'
        )
        ep_summary['atari_stat_op'] = tf.summary.merge(
            [
                tf.summary.scalar('episode/total_reward', ep_summary['total_r']),
                tf.summary.scalar('episode/steps', ep_summary['steps'])
            ],
            name='episode_atari'
        )
        self.log.debug('model-wide and episode summaries ok.')
        return model_summary, ep_summary

    def _make_runners(self, policy):
        """
        Defines thread-runners processes instances.

        Args:
            policy:     policy for runner to execute

        Returns:
            list of runners
        """
        # Replay memory_config:
        if self.use_memory:
            memory_config = dict(
                class_ref=Memory,
                kwargs=dict(
                    history_size=self.replay_memory_size,
                    max_sample_size=self.replay_rollout_length,
                    priority_sample_size=self.rp_sequence_size,
                    reward_threshold=self.rp_reward_threshold,
                    use_priority_sampling=self.use_reward_prediction,
                    task=self.task,
                    log_level=self.log_level,
                )
            )
        else:
            memory_config = None

        # Make runners:
        # `rollout_length` represents the number of "local steps":  the number of time steps
        # we run the policy before we get full rollout, run train step and update the parameters.
        runners = []
        task = 0  # Runners will have [worker_task][env_count] id's
        for env in self.env_list:
            kwargs=dict(
                env=env,
                policy=policy,
                task=self.task + task,
                rollout_length=self.rollout_length,  # ~20
                episode_summary_freq=self.episode_summary_freq,
                env_render_freq=self.env_render_freq,
                test=self.test_mode,
                ep_summary=self.ep_summary,
                memory_config=memory_config,
                log_level=self.log_level,
                global_step_op=self.global_step,
                aux_render_modes=self.aux_render_modes
            )
            kwargs.update(self.runner_config['kwargs'])
            # New runner instance:
            runners.append(self.runner_config['class_ref'](**kwargs))
            task += 0.01
        self.log.debug('runners ok.')
        return runners

    def _make_step_counters(self):
        """
        Defines operations for global step and global episode;

        Returns:
            None, sets attrs.
        """
        self.global_step = tf.get_variable(
            "global_step",
            [],
            tf.int32,
            initializer=tf.constant_initializer(
                0,
                dtype=tf.int32
            ),
            trainable=False
        )
        tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step)
        self.reset_global_step = self.global_step.assign(0)

        self.global_episode = tf.get_variable(
            "global_episode",
            [],
            tf.int32,
            initializer=tf.constant_initializer(
                0,
                dtype=tf.int32
            ),
            trainable=False
        )
        # Increment episode count:
        self.inc_episode = self.global_episode.assign_add(1)

    def _make_policy(self, scope):
        """
        Configures and instantiates policy network and ops.

        Note:
            `global` name_scope networks should be defined first.

        Args:
            scope:  name scope

        Returns:
            policy instance
        """
        with tf.variable_scope(scope):
            # Make policy instance:
            network = self.policy_class(**self.policy_kwargs)
            if 'global' not in scope:
                try:
                    # For locals those should be already defined:
                    assert hasattr(self, 'global_step') and \
                           hasattr(self, 'global_episode') and \
                           hasattr(self, 'inc_episode')
                    # Add attrs to local:
                    network.global_step = self.global_step
                    network.global_episode = self.global_episode
                    network.inc_episode= self.inc_episode
                    # Override with aac method:
                    network.get_sample_config = self.get_sample_config

                except AssertionError:
                    self.log.exception(
                        '`global` name_scope network[s] should be defined before any `local` one[s].'.
                        format(self.task)
                    )
                    raise RuntimeError
            else:
                # Set counters:
                self._make_step_counters()

        return network

    def _make_dummy_policy(self):
        class _Dummy(object):
            """
            Policy plug when target network is not used.
            """
            def __init__(self):
                self.on_state_in = None
                self.off_state_in = None
                self.on_lstm_state_pl_flatten = None
                self.off_lstm_state_pl_flatten = None
                self.on_a_r_in = None
                self.off_a_r_in = None
                self.on_logits = None
                self.off_logits = None
                self.on_vf = None
                self.off_vf = None
                self.on_batch_size = None
                self.on_time_length = None
                self.off_batch_size = None
                self.off_time_length = None
        return _Dummy()

    def get_data(self, **kwargs):
        """
        Collect rollouts from every environment.

        Returns:
            dictionary of lists of data streams collected from every runner
        """
        data_streams = [get_it(**kwargs) for get_it in self.data_getter]

        return {key: [stream[key] for stream in data_streams] for key in data_streams[0].keys()}

    def get_sample_config(self, _new_trial=True, **kwargs):
        """
        WARNING: _new_trial=True is quick fix, TODO: fix it properly!
        Returns environment configuration parameters for next episode to sample.
        By default is simple stateful iterator,
        works correctly with `DTGymDataset` data class, repeating cycle:
            - sample `num_train_episodes` from train data,
            - sample `num_test_episodes` from test data.

        Convention: supposed to override dummy method of local policy instance, see inside ._make_policy() method

        Returns:
            configuration dictionary of type `btgym.datafeed.base.EnvResetConfig`
        """
        # sess = tf.get_default_session()
        if self.current_train_episode < self.num_train_episodes:
            episode_type = 0  # train
            self.current_train_episode += 1
            self.log.debug(
                'c_1, c_train={}, c_test={}, type={}'.
                format(self.current_train_episode, self.current_test_episode, episode_type)
            )
        else:
            if self.current_test_episode < self.num_test_episodes:
                episode_type = 1  # test
                self.current_test_episode += 1
                self.log.debug(
                    'c_2, c_train={}, c_test={}, type={}'.
                    format(self.current_train_episode, self.current_test_episode, episode_type)
                )
            else:
                # cycle end, reset and start new (rec. depth 1)
                self.current_train_episode = 0
                self.current_test_episode = 0
                self.log.debug(
                    'c_3, c_train={}, c_test={}'.
                    format(self.current_train_episode, self.current_test_episode)
                )
                return self.get_sample_config(_new_trial=True)

        # Compose btgym.datafeed.base.EnvResetConfig-consistent dict:
        sample_config = dict(
            episode_config=dict(
                get_new=True,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            ),
            trial_config=dict(
                get_new=_new_trial,
                sample_type=episode_type,
                b_alpha=1.0,
                b_beta=1.0
            )
        )
        return sample_config

    def start(self, sess, summary_writer, **kwargs):
        """
        Executes all initializing operations,
        starts environment runner[s].
        Supposed to be called by parent worker just before training loop starts.

        Args:
            sess:           tf session object.
            kwargs:         not used by default.
        """
        try:
            # Copy weights from global to local:
            sess.run(self.sync)

            # Start thread_runners:
            self._start_runners(sess, summary_writer, **kwargs)

        except Exception as e:
            msg = 'start() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise e

    def _start_runners(self, sess, summary_writer, **kwargs):
        """

        Args:
            sess:
            summary_writer:

        Returns:

        """
        for runner in self.runners:
            runner.start_runner(sess, summary_writer, **kwargs)  # starting runner threads

        self.summary_writer = summary_writer

    def _get_rp_feeder(self, pi, batch):
        """
        Returns feed dictionary for `reward prediction` loss estimation subgraph.

        Args:
            pi:     policy to feed
        """
        feeder = feed_dict_from_nested(pi.rp_state_in, batch['state'])
        feeder.update(
            {
                pi.rp_target: batch['rp_target'],
                pi.rp_batch_size: batch['batch_size'],
            }
        )
        return feeder

    def _get_vr_feeder(self, pi, batch):
        """
        Returns feed dictionary for `value replay` loss estimation subgraph.

        Args:
            pi:     policy to feed
        """
        if not self.use_off_policy_aac:  # use single pass of network on same off-policy batch
            feeder = feed_dict_from_nested(pi.vr_state_in, batch['state'])
            feeder.update(feed_dict_rnn_context(pi.vr_lstm_state_pl_flatten, batch['context']))
            feeder.update(
                {
                    pi.vr_batch_size: batch['batch_size'],
                    pi.vr_time_length: batch['time_steps'],
                    pi.vr_last_a_in: batch['last_action'],
                    pi.vr_last_reward_in: batch['last_reward'],
                    pi.vr_target: batch['r']
                }
            )
        else:
            feeder = {pi.vr_target: batch['r']}  # redundant actually :)
        return feeder

    def _get_pc_feeder(self, pi, batch):
        """
        Returns feed dictionary for `pixel control` loss estimation subgraph.

        Args:
            pi:     policy to feed
        """
        if not self.use_off_policy_aac:  # use single pass of network on same off-policy batch
            feeder = feed_dict_from_nested(pi.pc_state_in, batch['state'])
            feeder.update(
                feed_dict_rnn_context(pi.pc_lstm_state_pl_flatten, batch['context']))
            feeder.update(
                {
                    pi.pc_last_a_in: batch['last_action'],
                    pi.pc_last_reward_in: batch['last_reward'],
                    pi.pc_action: batch['action'],
                    pi.pc_target: batch['pixel_change']
                }
            )
        else:
            feeder = {pi.pc_action: batch['action'], pi.pc_target: batch['pixel_change']}
        return feeder

    def _process_rollouts(self, rollouts):
        """
        rollout.process wrapper: makes single batch from list of rollouts

        Args:
            rollouts:   list of btgym.algorithms.Rollout class instances

        Returns:
            single batch data

        """
        batch = batch_stack(
            [
                r.process(
                    gamma=self.model_gamma,
                    gae_lambda=self.model_gae_lambda,
                    size=self.rollout_length,
                    time_flat=self.time_flat,
                ) for r in rollouts
            ]
        )
        return batch

    def _get_main_feeder(
            self,
            sess,
            on_policy_batch=None,
            off_policy_batch=None,
            rp_batch=None,
            is_train=True,
            pi=None,
            pi_prime=None):
        """
        Composes entire train step feed dictionary.
        Args:
            sess:                   tf session obj.
            pi:                     policy to feed
            pi_prime:               optional policy to feed
            on_policy_batch:        on-policy data batch
            off_policy_batch:       off-policy (replay memory) data batch
            rp_batch:               off-policy reward prediction data batch
            is_train (bool):        is data provided are train or test

        Returns:
            feed_dict (dict):   train step feed dictionary
        """
        feed_dict = {}
        # Feeder for on-policy AAC loss estimation graph:
        if on_policy_batch is not None:
            feed_dict = feed_dict_from_nested(pi.on_state_in, on_policy_batch['state'])
            feed_dict.update(
                feed_dict_rnn_context(pi.on_lstm_state_pl_flatten, on_policy_batch['context'])
            )
            feed_dict.update(
                {
                    pi.on_last_a_in: on_policy_batch['last_action'],
                    pi.on_last_reward_in: on_policy_batch['last_reward'],
                    pi.on_batch_size: on_policy_batch['batch_size'],
                    pi.on_time_length: on_policy_batch['time_steps'],
                    pi.on_pi_act_target: on_policy_batch['action'],
                    pi.on_pi_adv_target: on_policy_batch['advantage'],
                    pi.on_pi_r_target: on_policy_batch['r'],
                    pi.train_phase: is_train,  # Zeroes learn rate, [+ batch_norm + dropout]
                }
            )
            if self.use_target_policy and pi_prime is not None:
                feed_dict.update(
                    feed_dict_from_nested(pi_prime.on_state_in, on_policy_batch['state'])
                )
                feed_dict.update(
                    feed_dict_rnn_context(pi_prime.on_lstm_state_pl_flatten, on_policy_batch['context'])
                )
                feed_dict.update(
                    {
                        pi_prime.on_batch_size: on_policy_batch['batch_size'],
                        pi_prime.on_time_length: on_policy_batch['time_steps'],
                        pi_prime.on_last_a_in: on_policy_batch['last_action'],
                        pi_prime.on_last_reward_in: on_policy_batch['last_reward'],
                        # TODO: pi prime train phase?
                    }
                )
        if (self.use_any_aux_tasks or self.use_off_policy_aac) and off_policy_batch is not None:
            # Feeder for off-policy AAC loss estimation graph:
            off_policy_feed_dict = feed_dict_from_nested(pi.off_state_in, off_policy_batch['state'])
            off_policy_feed_dict.update(
                feed_dict_rnn_context(pi.off_lstm_state_pl_flatten, off_policy_batch['context']))
            off_policy_feed_dict.update(
                {
                    pi.off_last_a_in: off_policy_batch['last_action'],
                    pi.off_last_reward_in: off_policy_batch['last_reward'],
                    pi.off_batch_size: off_policy_batch['batch_size'],
                    pi.off_time_length: off_policy_batch['time_steps'],
                    pi.off_pi_act_target: off_policy_batch['action'],
                    pi.off_pi_adv_target: off_policy_batch['advantage'],
                    pi.off_pi_r_target: off_policy_batch['r'],
                }
            )
            if self.use_target_policy and pi_prime is not None:
                off_policy_feed_dict.update(
                    feed_dict_from_nested(pi_prime.off_state_in, off_policy_batch['state'])
                )
                off_policy_feed_dict.update(
                    {
                        pi_prime.off_batch_size: off_policy_batch['batch_size'],
                        pi_prime.off_time_length: off_policy_batch['time_steps'],
                        pi_prime.off_last_a_in: off_policy_batch['last_action'],
                        pi_prime.off_last_reward_in: off_policy_batch['last_reward'],
                    }
                )
                off_policy_feed_dict.update(
                    feed_dict_rnn_context(
                        pi_prime.off_lstm_state_pl_flatten,
                        off_policy_batch['context']
                    )
                )
            feed_dict.update(off_policy_feed_dict)

            # Update with reward prediction subgraph:
            if self.use_reward_prediction and rp_batch is not None:
                # Rebalanced 50/50 sample for RP:
                feed_dict.update(self._get_rp_feeder(pi, rp_batch))

            # Pixel control ...
            if self.use_pixel_control and off_policy_batch is not None:
                feed_dict.update(self._get_pc_feeder(pi, off_policy_batch))

            # VR...
            if self.use_value_replay and off_policy_batch is not None:
                feed_dict.update(self._get_vr_feeder(pi, off_policy_batch))

        return feed_dict

    def process_data(self, sess, data, is_train, pi, pi_prime=None):
        """
        Processes data, composes train step feed dictionary.
        Args:
            sess:               tf session obj.
            pi:                 policy to feed
            pi_prime:           optional policy to feed
            data (dict):        data dictionary
            is_train (bool):    is data provided are train or test

        Returns:
            feed_dict (dict):   train step feed dictionary
        """
        # Process minibatch for on-policy train step:
        on_policy_batch = self._process_rollouts(data['on_policy'])

        if self.use_memory:
            # Process rollouts from replay memory:
            off_policy_batch = self._process_rollouts(data['off_policy'])

            if self.use_reward_prediction:
                # Rebalanced 50/50 sample for RP:
                rp_rollouts = data['off_policy_rp']
                rp_batch = batch_stack([rp.process_rp(self.rp_reward_threshold) for rp in rp_rollouts])

            else:
                rp_batch = None

        else:
            off_policy_batch = None
            rp_batch = None

        return self._get_main_feeder(sess, on_policy_batch, off_policy_batch, rp_batch, is_train, pi, pi_prime)

    def process_summary(self, sess, data, model_data=None, step=None, episode=None):
        """
        Fetches and writes summary data from `data` and `model_data`.
        Args:
            sess:               tf summary obj.
            data(dict):         thread_runner rollouts and metadata
            model_data(dict):   model summary data
            step:               int, global step or None
            episode:            int, global episode number or None
        """
        if step is None:
            step = sess.run(self.global_step)

        if episode is None:
            episode = sess.run(self.global_episode)
        # Every worker writes train episode summaries:
        ep_summary_feeder = {}

        # Look for train episode summaries from all env runners:

        # self.log.warning('data+ep_summary: {}'.format( data['ep_summary']))

        for stat in data['ep_summary']:
            if stat is not None:
                for key in stat.keys():
                    if key in ep_summary_feeder.keys():
                        ep_summary_feeder[key] += [stat[key]]
                    else:
                        ep_summary_feeder[key] = [stat[key]]

        # Average values among thread_runners, if any, and write episode summary:

        # self.log.warning('ep_summary_feeder: {}'.format(ep_summary_feeder))

        if ep_summary_feeder != {}:
            ep_summary_feed_dict = {
                self.ep_summary[key]: np.average(list) for key, list in ep_summary_feeder.items()
            }

            if self.test_mode:
                # Atari:
                fetched_episode_stat = sess.run(self.ep_summary['atari_stat_op'], ep_summary_feed_dict)

            else:
                # BTGym
                fetched_episode_stat = sess.run(self.ep_summary['btgym_stat_op'], ep_summary_feed_dict)

            self.summary_writer.add_summary(fetched_episode_stat, episode)
            # self.summary_writer.flush()

        # Every worker writes test episode  summaries:
        test_ep_summary_feeder = {}

        # Look for test episode summaries:

        # self.log.warning('data+test_ep_summary: {}'.format(data['test_ep_summary']))

        for stat in data['test_ep_summary']:
            if stat is not None:
                for key in stat.keys():
                    if key in test_ep_summary_feeder.keys():
                        test_ep_summary_feeder[key] += [stat[key]]
                    else:
                        test_ep_summary_feeder[key] = [stat[key]]

        # Average values among thread_runners, if any, and write episode summary:

        # self.log.warning('test_ep_summary_feeder: {}'.format(test_ep_summary_feeder))

        if test_ep_summary_feeder != {}:
            test_ep_summary_feed_dict = {
                self.ep_summary[key]: np.average(list) for key, list in test_ep_summary_feeder.items()
            }
            fetched_test_episode_stat = sess.run(self.ep_summary['test_btgym_stat_op'], test_ep_summary_feed_dict)
            self.summary_writer.add_summary(fetched_test_episode_stat, episode)

        # Look for renderings (chief worker only, always 0-numbered environment in a list):
        if self.task == 0:
            if data['render_summary'][0] is not None:

                #self.log.warning('data[render_summary]: {}'.format(data['render_summary']))
                #self.log.warning('self.ep_summary: {}'.format(self.ep_summary))

                render_feed_dict = {
                    self.ep_summary[key]: pic for key, pic in data['render_summary'][0].items()
                }
                renderings = sess.run(self.ep_summary['render_op'], render_feed_dict)
                self.summary_writer.add_summary(renderings, episode)
                self.summary_writer.flush()

        # Every worker writes train episode summaries:
        if model_data is not None:
            self.summary_writer.add_summary(tf.Summary.FromString(model_data), step)
        self.summary_writer.flush()

    def process(self, sess, **kwargs):
        """
        Main train step method wrapper. Override if needed.

        Args:
            sess (tensorflow.Session):   tf session obj.
            kwargs:                      any


        """
        # return self._process(sess)
        self._process(sess)

    def _process(self, sess):
        """
        Grabs an on_policy_rollout [and off_policy rollout[s] from replay memory] that's been produced
        by the thread runner. If data identified as 'train data' - computes gradients and updates the parameters;
        writes summaries if any. The update is then sent to the parameter server.
        If on_policy_rollout identified as 'test data' -  no policy update is performed (learn rate is set to zero);
        Note that test data does not get stored in replay memory (thread runner area).
        Writes all available summaries.

        Args:
            sess (tensorflow.Session):   tf session obj.
        """
        # Quick wrap to get direct traceback from this trainer if something goes wrong:
        try:
            # Collect data from child thread runners:
            data = self.get_data()

            # Copy weights from local policy to local target policy:
            if self.use_target_policy and self.local_steps % self.pi_prime_update_period == 0:
                sess.run(self.sync_pi_prime)

            # Test or train: if at least one on-policy rollout from parallel runners is test one -
            # set learn rate to zero for entire minibatch. Doh.
            try:
                is_train = not np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any()

            except KeyError:
                is_train = True

            self.log.debug(
                'Got rollout episode. type: {}, trial_type: {}, is_train: {}'.format(
                    np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any(),
                    np.asarray([env['state']['metadata']['trial_type'] for env in data['on_policy']]).any(),
                    is_train
                )
            )

            if is_train:
                # If there is no any test rollouts  - do a train step:
                sess.run(self.sync_pi)  # only sync at train time

                feed_dict = self.process_data(sess, data, is_train, self.local_network, self.local_network_prime)

                # Say `No` to redundant summaries:
                wirte_model_summary =\
                    self.local_steps % self.model_summary_freq == 0

                #fetches = [self.train_op, self.local_network.debug]  # include policy debug shapes
                fetches = [self.train_op]

                if wirte_model_summary:
                    fetches_last = fetches + [self.model_summary_op, self.inc_step]
                else:
                    fetches_last = fetches + [self.inc_step]

                # Do a number of SGD train epochs:
                # When doing more than one epoch, we actually use only last summary:
                for i in range(self.num_epochs - 1):
                    fetched = sess.run(fetches, feed_dict=feed_dict)

                fetched = sess.run(fetches_last, feed_dict=feed_dict)

                if wirte_model_summary:
                    model_summary = fetched[-2]

                else:
                    model_summary = None

                self.local_steps += 1  # only update on train steps

            else:
                model_summary = None

            # Write down summaries:
            self.process_summary(sess, data, model_summary)

            # print debug info:
            #for k, v in fetched[1].items():
            #    print('{}: {}'.format(k,v))
            #print('\n')

            #for k, v in feed_dict.items():
            #    try:
            #        print(k, v.shape)
            #    except:
            #        print(k, type(v))

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        except:
            msg = 'process() exception occurred' + \
                '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


class Unreal(BaseAAC):
    """
    Unreal: Asynchronous Advantage Actor Critic with auxiliary control tasks.

    Auxiliary tasks implementation borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
    https://miyosuda.github.io/
    https://github.com/miyosuda/unreal

    Original A3C code comes from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Papers:
    https://arxiv.org/abs/1602.01783
    https://arxiv.org/abs/1611.05397
    """
    def __init__(self, **kwargs):
        """
        See BaseAAC class args for details:

        Args:
            env:                    environment instance or list of instances
            task:                   int, parent worker id
            policy_config:          policy estimator class and configuration dictionary
            log_level:              int, logbook.level
            on_policy_loss:         callable returning tensor holding on_policy training loss graph and summaries
            off_policy_loss:        callable returning tensor holding off_policy training loss graph and summaries
            vr_loss:                callable returning tensor holding value replay loss graph and summaries
            rp_loss:                callable returning tensor holding reward prediction loss graph and summaries
            pc_loss:                callable returning tensor holding pixel_control loss graph and summaries
            random_seed:            int or None
            model_gamma:            scalar, gamma discount factor
            model_gae_lambda:       scalar, GAE lambda
            model_beta:             entropy regularization beta, scalar or [high_bound, low_bound] for log_uniform.
            opt_max_env_steps:      int, total number of environment steps to run training on.
            opt_decay_steps:        int, learn ratio decay steps, in number of environment steps.
            opt_end_learn_rate:     scalar, final learn rate
            opt_learn_rate:         start learn rate, scalar or [high_bound, low_bound] for log_uniform distr.
            opt_decay:              scalar, optimizer decay, if apll.
            opt_momentum:           scalar, optimizer momentum, if apll.
            opt_epsilon:            scalar, optimizer epsilon
            rollout_length:         int, on-policy rollout length
            time_flat:              bool, flatten rnn time-steps in rollouts while training - see `Notes` below
            episode_train_test_cycle:   tuple or list as (train_number, test_number), def=(1,0): enables infinite
                                        loop such as: run `train_number` of train data episodes,
                                        than `test_number` of test data episodes, repeat. Should be consistent
                                        with provided dataset parameters (test data should exist if `test_number > 0`)
            episode_summary_freq:   int, write episode summary for every i'th episode
            env_render_freq:        int, write environment rendering summary for every i'th train step
            model_summary_freq:     int, write model summary for every i'th train step
            test_mode:              bool, True: Atari, False: BTGym
            replay_memory_size:     int, in number of experiences
            replay_batch_size:      int, mini-batch size for off-policy training, def = 1
            replay_rollout_length:  int off-policy rollout length by def. equals on_policy_rollout_length
            use_off_policy_aac:     bool, use full AAC off-policy loss instead of Value-replay
            use_reward_prediction:  bool, use aux. off-policy reward prediction task
            use_pixel_control:      bool, use aux. off-policy pixel control task
            use_value_replay:       bool, use aux. off-policy value replay task (not used if use_off_policy_aac=True)
            rp_lambda:              reward prediction loss weight, scalar or [high, low] for log_uniform distr.
            pc_lambda:              pixel control loss weight, scalar or [high, low] for log_uniform distr.
            vr_lambda:              value replay loss weight, scalar or [high, low] for log_uniform distr.
            off_aac_lambda:         off-policy AAC loss weight, scalar or [high, low] for log_uniform distr.
            gamma_pc:               NOT USED
            rp_reward_threshold:    scalar, reward prediction classification threshold, above which reward is 'non-zero'
            rp_sequence_size:       int, reward prediction sample size, in number of experiences
            clip_epsilon:           scalar, PPO: surrogate L^clip epsilon
            num_epochs:             int, num. of SGD runs for every train step, val. > 1 should be used with caution.
            pi_prime_update_period: int, PPO: pi to pi_old update period in number of train steps, def: 1
            _use_target_policy:     bool, PPO: use target policy (aka pi_old), delayed by `pi_prime_update_period` delay

        Note:
            - On `time_flat` arg:

                There are two alternatives to run RNN part of policy estimator:

                a. Feed initial RNN state for every experience frame in rollout
                        (those are stored anyway if we want random memory repaly sampling) and do single time-step RNN
                        advance for all experiences in a batch; this is when time_flat=True;

                b. Reshape incoming batch after convolution part of network in time-wise fashion
                        for every rollout in a batch i.e. batch_size=number_of_rollouts and
                        rnn_timesteps=max_rollout_length. In this case we need to feed initial rnn_states
                        for rollouts only. There is some little extra work to pad rollouts to max_time_size
                        and feed true rollout lengths to rnn. Thus, when time_flat=False, we unroll RNN in
                        specified number of time-steps for every rollout.

                Both options has pros and cons:

                Unrolling dynamic RNN is computationally more expensive but gives clearly faster convergence,
                    [possibly] due to the fact that RNN states for 2nd, 3rd, ... frames
                    of rollouts are computed using updated policy estimator, which is supposed to be
                    closer to optimal one. When time_flattened, every time-step uses RNN states computed
                    when rollout was collected (i.e. by behavioral policy estimator with older
                    parameters).

                Nevertheless, time_flatting can be interesting
                    because one can safely shuffle training batch or mix on-policy and off-policy data in single mini-batch,
                    ensuring iid property and allowing, say, proper batch normalisation (this has yet to be tested).
        """
        try:
            super(Unreal, self).__init__(name='UNREAL', **kwargs)
        except:
            msg = 'Child class Unreal __init()__ exception occurred' + \
                  '\n\nPress `Ctrl-C` or jupyter:[Kernel]->[Interrupt] for clean exit.\n'
            self.log.exception(msg)
            raise RuntimeError(msg)


class A3C(BaseAAC):
    """
    Vanilla Asynchronous Advantage Actor Critic algorithm.

    Based on original code taken from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent

    Paper: https://arxiv.org/abs/1602.01783
    """

    def __init__(self, **kwargs):
        """
        A3C args. is a subset of BaseAAC arguments, see `BaseAAC` class for descriptions.

        Args:
            env:
            task:
            policy_config:
            log:
            random_seed:
            model_gamma:
            model_gae_lambda:
            model_beta:
            opt_max_env_steps:
            opt_decay_steps:
            opt_end_learn_rate:
            opt_learn_rate:
            opt_decay:
            opt_momentum:
            opt_epsilon:
            rollout_length:
            episode_summary_freq:
            env_render_freq:
            model_summary_freq:
            test_mode:
        """
        super(A3C, self).__init__(
            on_policy_loss=aac_loss_def,
            use_off_policy_aac=False,
            use_reward_prediction=False,
            use_pixel_control=False,
            use_value_replay=False,
            _use_target_policy=False,
            name='A3C',
            **kwargs
        )


class PPO(BaseAAC):
    """
    AAC with Proximal Policy Optimization surrogate L^Clip loss,
    optionally augmented with auxiliary control tasks.

    paper:
    https://arxiv.org/pdf/1707.06347.pdf

    Based on PPO-SGD code from OpenAI `Baselines` repository under MIT licence:
    https://github.com/openai/baselines

    Async. framework code comes from OpenAI repository under MIT licence:
    https://github.com/openai/universe-starter-agent
    """
    def __init__(self, **kwargs):
        """
         PPO args. is a subset of BaseAAC arguments, see `BaseAAC` class for descriptions.

        Args:
            env:
            task:
            policy_config:
            log_level:
            vr_loss:
            rp_loss:
            pc_loss:
            random_seed:
            model_gamma:
            model_gae_lambda:
            model_beta:
            opt_max_env_steps:
            opt_decay_steps:
            opt_end_learn_rate:
            opt_learn_rate:
            opt_decay:
            opt_momentum:
            opt_epsilon:
            rollout_length:
            episode_summary_freq:
            env_render_freq:
            model_summary_freq:
            test_mode:
            replay_memory_size:
            replay_rollout_length:
            use_off_policy_aac:
            use_reward_prediction:
            use_pixel_control:
            use_value_replay:
            rp_lambda:
            pc_lambda:
            vr_lambda:
            off_aac_lambda:
            rp_reward_threshold:
            rp_sequence_size:
            clip_epsilon:
            num_epochs:
            pi_prime_update_period:
        """
        super(PPO, self).__init__(
            on_policy_loss=ppo_loss_def,
            off_policy_loss=ppo_loss_def,
            _use_target_policy=True,
            name='PPO',
            **kwargs
        )


