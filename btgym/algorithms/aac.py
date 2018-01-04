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

from logbook import Logger, StreamHandler
import sys

import numpy as np
import tensorflow as tf

from btgym.spaces import DictSpace as ObSpace  # now can simply be gym.Dict
from btgym.algorithms import Memory, make_data_getter, RunnerThread
from btgym.algorithms.math_utils import log_uniform
from btgym.algorithms.losses import value_fn_loss_def, rp_loss_def, pc_loss_def, aac_loss_def, ppo_loss_def, state_min_max_loss_def
from btgym.algorithms.utils import feed_dict_rnn_context, feed_dict_from_nested, batch_stack


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
                 on_policy_loss=aac_loss_def,
                 off_policy_loss=aac_loss_def,
                 vr_loss=value_fn_loss_def,
                 rp_loss=rp_loss_def,
                 pc_loss=pc_loss_def,
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
                 _use_target_policy=False):  # target policy tracking behavioral one with delay
        """

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
        # Logging:
        self.log_level = log_level
        self.task = task
        StreamHandler(sys.stdout).push_application()
        self.log = Logger('AAC_{}'.format(self.task), level=self.log_level)

        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)
        self.log.debug('rnd_seed:{}, log_u_sample_(0,1]x5: {}'.
                       format(random_seed, log_uniform([1e-10,1], 5)))

        self.env_list = env
        try:
            assert isinstance(self.env_list, list)

        except AssertionError:
            self.env_list = [env]

        ref_env = self.env_list[0]  # reference instance to get obs shapes etc.
        assert isinstance(ref_env.observation_space, ObSpace),\
            'expected environment observation space of type {}, got: {}'.\
            format(ObSpace, type(ref_env.observation_space))


        self.policy_class = policy_config['class_ref']
        self.policy_kwargs = policy_config['kwargs']

        # Losses:
        self.on_policy_loss = on_policy_loss
        self.off_policy_loss = off_policy_loss
        self.vr_loss = vr_loss
        self.rp_loss = rp_loss
        self.pc_loss = pc_loss

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
        self.use_memory = self.use_any_aux_tasks or self.use_off_policy_aac

        self.use_target_policy = _use_target_policy

        self.log.warning('learn_rate: {:1.6f}, entropy_beta: {:1.6f}'.format(self.opt_learn_rate, self.model_beta))

        if self.use_off_policy_aac:
            self.log.warning('off_aac_lambda: {:1.6f}'.format(self.off_aac_lambda,))

        if self.use_any_aux_tasks:
            self.log.warning('vr_lambda: {:1.6f}, pc_lambda: {:1.6f}, rp_lambda: {:1.6f}'.
                          format(self.vr_lambda, self.pc_lambda, self.rp_lambda))


        #self.log.info(
        #    'AAC_{}: max_steps: {}, decay_steps: {}, end_rate: {:1.6f},'.
        #        format(self.task, self.opt_max_env_steps, self.opt_decay_steps, self.opt_end_learn_rate))

        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        # Update policy configuration
        self.policy_kwargs.update(
            {
                'ob_space': ref_env.observation_space.shape,
                'ac_space': ref_env.action_space.n,
                'rp_sequence_size': self.rp_sequence_size,
                'aux_estimate': self.use_any_aux_tasks,
            }
        )
        # Start building graphs:
        self.log.debug('started building graphs')
        # PS:
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            self.network = self.make_policy('global')

        # Worker:
        with tf.device(worker_device):
            self.local_network = pi = self.make_policy('local')

            if self.use_target_policy:
                self.local_network_prime = pi_prime = self.make_policy('local_prime')

            else:
                self.local_network_prime = pi_prime = self._make_dummy_policy()

            # Meant for Batch-norm layers:
            pi.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*local.*')

            self.log.debug('local_network_upd_ops_collection:\n{}'.format(pi.update_ops))
            self.log.debug('\nlocal_network_var_list_to_save:')
            for v in pi.var_list:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            #  Learning rate annealing:
            learn_rate_decayed = tf.train.polynomial_decay(
                self.opt_learn_rate,
                self.global_step + 1,
                self.opt_decay_steps,
                self.opt_end_learn_rate,
                power=1,
                cycle=False,
            )
            clip_epsilon = tf.cast(self.clip_epsilon * learn_rate_decayed / self.opt_learn_rate, tf.float32)

            # Freeze training if train_phase is False:
            train_learn_rate = learn_rate_decayed * tf.cast(pi.train_phase, tf.float64)
            self.log.debug('learn rate ok')

            # On-policy AAC loss definition:
            self.on_pi_act_target = tf.placeholder(tf.float32, [None, ref_env.action_space.n], name="on_policy_action_pl")
            self.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
            self.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")

            on_pi_loss, on_pi_summaries = self.on_policy_loss(
                act_target=self.on_pi_act_target,
                adv_target=self.on_pi_adv_target,
                r_target=self.on_pi_r_target,
                pi_logits=pi.on_logits,
                pi_vf=pi.on_vf,
                pi_prime_logits=pi_prime.on_logits,
                entropy_beta=self.model_beta,
                epsilon=clip_epsilon,
                name='on_policy',
                verbose=True
            )
            # Start accumulating total loss:
            self.loss = on_pi_loss
            model_summaries = on_pi_summaries

            # wrong EXPERIMENT:
            if False:
                min_max_loss, min_max_summaries = state_min_max_loss_def(
                    ohlc_targets=pi.raw_state,
                    min_max_state=pi.state_min_max,
                    name='on_policy',
                    verbose=True
                )
                self.loss = self.loss + 0.1 * min_max_loss
                model_summaries += min_max_summaries

            # Off-policy losses:
            self.off_pi_act_target = tf.placeholder(
                tf.float32, [None, ref_env.action_space.n], name="off_policy_action_pl")
            self.off_pi_adv_target = tf.placeholder(tf.float32, [None], name="off_policy_advantage_pl")
            self.off_pi_r_target = tf.placeholder(tf.float32, [None], name="off_policy_return_pl")

            if self.use_off_policy_aac:
                # Off-policy AAC loss graph mirrors on-policy:
                off_pi_loss, off_pi_summaries = self.off_policy_loss(
                    act_target=self.off_pi_act_target,
                    adv_target=self.off_pi_adv_target,
                    r_target=self.off_pi_r_target,
                    pi_logits=pi.off_logits,
                    pi_vf=pi.off_vf,
                    pi_prime_logits=pi_prime.off_logits,
                    entropy_beta=self.model_beta,
                    epsilon=clip_epsilon,
                    name='off_policy',
                    verbose=False
                )
                self.loss = self.loss + self.off_aac_lambda * off_pi_loss
                model_summaries += off_pi_summaries

            if self.use_pixel_control:
                # Pixel control loss:
                self.pc_action = tf.placeholder(tf.float32, [None, ref_env.action_space.n], name="pc_action")
                self.pc_target = tf.placeholder(tf.float32, [None, None, None], name="pc_target")

                pc_loss, pc_summaries = self.pc_loss(
                    actions=self.pc_action,
                    targets=self.pc_target,
                    pi_pc_q=pi.pc_q,
                    name='off_policy',
                    verbose=True
                )
                self.loss = self.loss + self.pc_lambda * pc_loss
                # Add specific summary:
                model_summaries += pc_summaries

            if self.use_value_replay:
                # Value function replay loss:
                self.vr_target = tf.placeholder(tf.float32, [None], name="vr_target")
                vr_loss, vr_summaries = self.vr_loss(
                    r_target=self.vr_target,
                    pi_vf=pi.vr_value,
                    name='off_policy',
                    verbose=True
                )
                self.loss = self.loss + self.vr_lambda * vr_loss
                model_summaries += vr_summaries

            if self.use_reward_prediction:
                # Reward prediction loss:
                self.rp_target = tf.placeholder(tf.float32, [None, 3], name="rp_target")

                rp_loss, rp_summaries = self.rp_loss(
                    rp_targets=self.rp_target,
                    pi_rp_logits=pi.rp_logits,
                    name='off_policy',
                    verbose=True
                )
                self.loss = self.loss + self.rp_lambda * rp_loss
                model_summaries += rp_summaries

            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # Copy weights from the parameter server to the local model
            self.sync = self.sync_pi = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            if self.use_target_policy:
                # Copy weights from new policy model to target one:
                self.sync_pi_prime = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi_prime.var_list, pi.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))

            # Since every observation mod. has same batch size - just take first key in a row:
            self.inc_step = self.global_step.assign_add(tf.shape(pi.on_state_in[list(pi.on_state_in.keys())[0]])[0])

            # Each worker gets a different set of adam optimizer parameters:
            opt = tf.train.AdamOptimizer(train_learn_rate, epsilon=1e-5)

            #opt = tf.train.RMSPropOptimizer(
            #    learning_rate=train_learn_rate,
            #    decay=self.opt_decay,
            #    momentum=self.opt_momentum,
            #    epsilon=self.opt_epsilon,
            #)

            #self.train_op = tf.group(*pi.update_ops, opt.apply_gradients(grads_and_vars), self.inc_step)
            #self.train_op = tf.group(opt.apply_gradients(grads_and_vars), self.inc_step)
            self.train_op = opt.apply_gradients(grads_and_vars)

            # Add model-wide statistics:
            with tf.name_scope('model'):
                model_summaries += [
                    tf.summary.scalar("grad_global_norm", tf.global_norm(grads)),
                    tf.summary.scalar("var_global_norm", tf.global_norm(pi.var_list)),
                    tf.summary.scalar("learn_rate", learn_rate_decayed),  # cause actual rate is a jaggy due to testing
                    tf.summary.scalar("total_loss", self.loss),
                ]

            self.summary_writer = None
            self.local_steps = 0

            self.log.debug('train op defined')

            # Model stat. summary:
            self.model_summary_op = tf.summary.merge(model_summaries, name='model_summary')

            # Episode-related summaries:
            self.ep_summary = dict(
                # Summary placeholders
                render_atari=tf.placeholder(tf.uint8, [None, None, None, 1]),
                total_r=tf.placeholder(tf.float32, ),
                cpu_time=tf.placeholder(tf.float32, ),
                final_value=tf.placeholder(tf.float32, ),
                steps=tf.placeholder(tf.int32, ),
            )

            if self.test_mode:
                # For Atari:
                self.ep_summary['render_op'] = tf.summary.image("model/state", self.ep_summary['render_atari'])

            else:
                # BTGym rendering:
                self.ep_summary.update(
                    {
                        mode: tf.placeholder(tf.uint8, [None, None, None, 3]) for mode in self.env_list[0].render_modes
                    }
                )
                self.ep_summary['render_op'] = tf.summary.merge(
                    [tf.summary.image(mode, self.ep_summary[mode]) for mode in self.env_list[0].render_modes]
                )

            # Episode stat. summary:
            self.ep_summary['btgym_stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode_train/total_reward', self.ep_summary['total_r']),
                    tf.summary.scalar('episode_train/cpu_time_sec', self.ep_summary['cpu_time']),
                    tf.summary.scalar('episode_train/final_value', self.ep_summary['final_value']),
                    tf.summary.scalar('episode_train/env_steps', self.ep_summary['steps'])
                ],
                name='episode_train_btgym'
            )
            # Test episode stat. summary:
            self.ep_summary['test_btgym_stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode_test/total_reward', self.ep_summary['total_r']),
                    tf.summary.scalar('episode_test/final_value', self.ep_summary['final_value']),
                    tf.summary.scalar('episode_test/env_steps', self.ep_summary['steps'])
                ],
                name='episode_test_btgym'
            )
            self.ep_summary['atari_stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode/total_reward', self.ep_summary['total_r']),
                    tf.summary.scalar('episode/steps', self.ep_summary['steps'])
                ],
                name='episode_atari'
            )

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
                        log=self.log,
                    )
                )
            else:
                memory_config = None

            # Make runners:
            # `rollout_length` represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            self.runners = []
            task = 100 * self.task  # Runners will have [worker_task][env_count] id's
            for env in self.env_list:
                self.runners.append(
                     RunnerThread(
                        env,
                        pi,
                        task,
                        self.rollout_length,  # ~20
                        self.episode_summary_freq,
                        self.env_render_freq,
                        self.test_mode,
                        self.ep_summary,
                        memory_config
                     )
                )
                task += 1
            # Make rollouts provider:
            self.data_getter = [make_data_getter(runner.queue) for runner in self.runners]

            self.log.debug('.init() done')

    def get_data(self):
        """
        Collect rollouts from every environmnet.

        Returns:
            dictionary of lists of data streams collected from every runner
        """
        # TODO: nowait?
        data_streams = [get_it() for get_it in self.data_getter]

        return {key: [stream[key] for stream in data_streams] for key in data_streams[0].keys()}

    def make_policy(self, scope):
        """
        Configures and instantiates policy network and ops.

        Note:
            `global` name_scope network should be defined first.

        Args:
            scope:  name scope

        Returns:
            policy instance
        """
        with tf.variable_scope(scope):
            # Make policy instance:
            network = self.policy_class(**self.policy_kwargs)
            if scope not in 'global':
                try:
                    # For locals those should be already defined:
                    assert hasattr(self, 'global_step') and \
                           hasattr(self, 'global_episode') and \
                           hasattr(self, 'inc_episode')
                    # Set for local:
                    network.global_step = self.global_step
                    network.global_episode = self.global_episode
                    network.inc_episode= self.inc_episode
                except:
                    raise AttributeError(
                        'AAC_{}: `global` name_scope network should be defined before any `local`s.'.
                        format(self.task)
                    )
            else:
                # Set counters:
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

    def start(self, sess, summary_writer):
        for runner in self.runners:
            runner.start_runner(sess, summary_writer)  # starting runner threads

        self.summary_writer = summary_writer

    def get_rp_feeder(self, batch):
        """
        Returns feed dictionary for `reward prediction` loss estimation subgraph.
        """
        feeder = feed_dict_from_nested(self.local_network.rp_state_in, batch['state'])
        feeder.update(
            {
                self.rp_target: batch['rp_target'],
                self.local_network.rp_batch_size: batch['batch_size'],
            }
        )
        return feeder

    def get_vr_feeder(self, batch):
        """
        Returns feed dictionary for `value replay` loss estimation subgraph.
        """
        if not self.use_off_policy_aac:  # use single pass of network on same off-policy batch
            feeder = feed_dict_from_nested(self.local_network.vr_state_in, batch['state'])
            feeder.update(feed_dict_rnn_context(self.local_network.vr_lstm_state_pl_flatten, batch['context']))
            feeder.update(
                {
                    self.local_network.vr_batch_size: batch['batch_size'],
                    self.local_network.vr_time_length: batch['time_steps'],
                    self.local_network.vr_a_r_in: batch['last_action_reward'],
                    self.vr_target: batch['r']
                }
            )
        else:
            feeder = {self.vr_target: batch['r']}  # redundant actually :)
        return feeder

    def get_pc_feeder(self, batch):
        """
        Returns feed dictionary for `pixel control` loss estimation subgraph.
        """
        if not self.use_off_policy_aac:  # use single pass of network on same off-policy batch
            feeder = feed_dict_from_nested(self.local_network.pc_state_in, batch['state'])
            feeder.update(
                feed_dict_rnn_context(self.local_network.pc_lstm_state_pl_flatten, batch['context']))
            feeder.update(
                {
                    self.local_network.pc_a_r_in: batch['last_action_reward'],
                    self.pc_action: batch['action'],
                    self.pc_target: batch['pixel_change']
                }
            )
        else:
            feeder = {self.pc_action: batch['action'], self.pc_target: batch['pixel_change']}
        return feeder

    def process(self, sess):
        """
        Grabs a on_policy_rollout that's been produced by the thread runner. If data identified as 'train data' -
        samples off_policy rollout[s] from replay memory and updates the parameters; writes summaries if any.
        The update is then sent to the parameter server.
        If on_policy_rollout contains 'test data' -  no policy update is performed and learn rate is set to zero;
        Meanwile test data are stored in replay memory.
        """

        # Collect data from child thread runners:
        data = self.get_data()

        # Test or train: if at least one rollout from parallel runners is test rollout -
        # set learn rate to zero for entire minibatch. Doh.
        try:
            is_train = not np.asarray([env['state']['metadata']['type'] for env in data['on_policy']]).any()

        except KeyError:
            is_train = True

        # Copy weights from local policy to local target policy:
        if self.use_target_policy and self.local_steps % self.pi_prime_update_period == 0:
            sess.run(self.sync_pi_prime)

        if is_train:
            # If there is no testing rollouts  - copy weights from shared to local new_policy:
            sess.run(self.sync_pi)

        #self.log.debug('is_train: {}'.format(is_train))

        # Process minibatch for on-policy train step:
        on_policy_rollouts = data['on_policy']
        on_policy_batch = batch_stack(
            [
                r.process(
                    gamma=self.model_gamma,
                    gae_lambda=self.model_gae_lambda,
                    size=self.rollout_length,
                    time_flat=self.time_flat,
                ) for r in on_policy_rollouts
            ]
        )
        # Feeder for on-policy AAC loss estimation graph:
        feed_dict = feed_dict_from_nested(self.local_network.on_state_in, on_policy_batch['state'])
        feed_dict.update(
            feed_dict_rnn_context(self.local_network.on_lstm_state_pl_flatten, on_policy_batch['context'])
        )
        feed_dict.update(
            {
                self.local_network.on_a_r_in: on_policy_batch['last_action_reward'],
                self.local_network.on_batch_size: on_policy_batch['batch_size'],
                self.local_network.on_time_length: on_policy_batch['time_steps'],
                self.on_pi_act_target: on_policy_batch['action'],
                self.on_pi_adv_target: on_policy_batch['advantage'],
                self.on_pi_r_target: on_policy_batch['r'],
                self.local_network.train_phase: is_train,  # Zeroes learn rate, [+ batch_norm]
            }
        )
        if self.use_target_policy:
            feed_dict.update(
                feed_dict_from_nested(self.local_network_prime.on_state_in, on_policy_batch['state'])
            )
            feed_dict.update(
                feed_dict_rnn_context(self.local_network_prime.on_lstm_state_pl_flatten, on_policy_batch['context'])
            )
            feed_dict.update(
                {
                    self.local_network_prime.on_batch_size: on_policy_batch['batch_size'],
                    self.local_network_prime.on_time_length: on_policy_batch['time_steps'],
                    self.local_network_prime.on_a_r_in: on_policy_batch['last_action_reward']
                }
            )
        if self.use_memory:
            # Process rollouts from replay memory:
            off_policy_rollouts = data['off_policy']
            off_policy_batch = batch_stack(
                [
                    r.process(
                        gamma=self.model_gamma,
                        gae_lambda=self.model_gae_lambda,
                        size=self.replay_rollout_length,
                        time_flat=self.time_flat,
                    ) for r in off_policy_rollouts
                ]
            )
            # Feeder for off-policy AAC loss estimation graph:
            off_policy_feed_dict = feed_dict_from_nested(self.local_network.off_state_in, off_policy_batch['state'])
            off_policy_feed_dict.update(
                feed_dict_rnn_context(self.local_network.off_lstm_state_pl_flatten, off_policy_batch['context']))
            off_policy_feed_dict.update(
                {
                    self.local_network.off_a_r_in: off_policy_batch['last_action_reward'],
                    self.local_network.off_batch_size: off_policy_batch['batch_size'],
                    self.local_network.off_time_length: off_policy_batch['time_steps'],
                    self.off_pi_act_target: off_policy_batch['action'],
                    self.off_pi_adv_target: off_policy_batch['advantage'],
                    self.off_pi_r_target: off_policy_batch['r'],
                }
            )
            if self.use_target_policy:
                off_policy_feed_dict.update(
                    feed_dict_from_nested(self.local_network_prime.off_state_in, off_policy_batch['state'])
                )
                off_policy_feed_dict.update(
                    {
                        self.local_network_prime.off_batch_size: off_policy_batch['batch_size'],
                        self.local_network_prime.off_time_length: off_policy_batch['time_steps'],
                        self.local_network_prime.off_a_r_in: off_policy_batch['last_action_reward']
                    }
                )
                off_policy_feed_dict.update(
                    feed_dict_rnn_context(
                        self.local_network_prime.off_lstm_state_pl_flatten,
                        off_policy_batch['context']
                    )
                )
            feed_dict.update(off_policy_feed_dict)

            # Update with reward prediction subgraph:
            if self.use_reward_prediction:
                # Rebalanced 50/50 sample for RP:
                rp_rollouts = data['off_policy_rp']
                rp_batch = batch_stack([rp.process_rp(self.rp_reward_threshold) for rp in rp_rollouts])
                feed_dict.update(self.get_rp_feeder(rp_batch))

            # Pixel control ...
            if self.use_pixel_control:
                feed_dict.update(self.get_pc_feeder(off_policy_batch))

            # VR...
            if self.use_value_replay:
                feed_dict.update(self.get_vr_feeder(off_policy_batch))

        # Every worker writes train episode and model summaries:
        ep_summary_feeder = {}

        # Look for train episode summaries from all env runners:
        for stat in data['ep_summary']:
            if stat is not None:
                for key in stat.keys():
                    if key in ep_summary_feeder.keys():
                        ep_summary_feeder[key] += [stat[key]]
                    else:
                        ep_summary_feeder[key] = [stat[key]]
        # Average values among thread_runners, if any, and write episode summary:
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

            self.summary_writer.add_summary(fetched_episode_stat, sess.run(self.global_episode))
            self.summary_writer.flush()

        # Every worker writes test episode  summaries:
        test_ep_summary_feeder = {}

        # Look for test episode summaries:
        for stat in data['test_ep_summary']:
            if stat is not None:
                for key in stat.keys():
                    if key in test_ep_summary_feeder.keys():
                        test_ep_summary_feeder[key] += [stat[key]]
                    else:
                        test_ep_summary_feeder[key] = [stat[key]]
                        # Average values among thread_runners, if any, and write episode summary:
            if test_ep_summary_feeder != {}:
                test_ep_summary_feed_dict = {
                    self.ep_summary[key]: np.average(list) for key, list in test_ep_summary_feeder.items()
                }
                fetched_test_episode_stat = sess.run(self.ep_summary['test_btgym_stat_op'], test_ep_summary_feed_dict)
                self.summary_writer.add_summary(fetched_test_episode_stat, sess.run(self.global_episode))
                self.summary_writer.flush()

        wirte_model_summary =\
            self.local_steps % self.model_summary_freq == 0

        # Look for renderings (chief worker only, always 0-numbered environment):
        if self.task == 0:
            if data['render_summary'][0] is not None:
                render_feed_dict = {
                    self.ep_summary[key]: pic for key, pic in data['render_summary'][0].items()
                }
                renderings = sess.run(self.ep_summary['render_op'], render_feed_dict)
                self.summary_writer.add_summary(renderings, sess.run(self.global_episode))
                self.summary_writer.flush()

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
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1

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

Unreal = BaseAAC


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
            **kwargs
        )


