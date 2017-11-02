

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten as flatten_nested

from btgym.spaces import BTgymMultiSpace
from btgym.algorithms import Memory, Rollout, make_rollout_getter, RunnerThread
from btgym.algorithms.math_util import log_uniform
from btgym.algorithms.losses import value_fn_loss_def, rp_loss_def, pc_loss_def, aac_loss_def
from btgym.algorithms.util import feed_dict_rnn_context, feed_dict_from_nested


class Unreal(object):
    """
    Asynchronous Advantage Actor Critic with auxiliary control tasks.

    This UNREAL implementation borrows heavily from Kosuke Miyoshi code, under Apache License 2.0:
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
                 log,
                 random_seed=None,
                 model_gamma=0.99,  # decay
                 model_gae_lambda=1.00,  # GAE lambda
                 model_beta=0.01,  # entropy regularizer
                 opt_max_train_steps=10**7,
                 opt_decay_steps=None,
                 opt_end_learn_rate=None,
                 opt_learn_rate=1e-4,
                 opt_decay=0.99,
                 opt_momentum=0.0,
                 opt_epsilon=1e-10,
                 rollout_length=20,
                 episode_summary_freq=2,  # every i`th environment episode
                 env_render_freq=10,  # every i`th environment episode
                 model_summary_freq=100,  # every i`th algorithm iteration
                 test_mode=False,  # gym_atari test mode
                 replay_memory_size=2000,
                 replay_rollout_length=None,
                 use_off_policy_aac=False,
                 use_reward_prediction=False,
                 use_pixel_control=False,
                 use_value_replay=False,
                 use_rebalanced_replay=False,  # simplified form of prioritized replay
                 rebalance_skewness=2,
                 rp_lambda=1,  # aux tasks loss weights
                 pc_lambda=0.1,
                 vr_lambda=1,
                 off_aac_lambda=1,
                 gamma_pc=0.9,  # pixel change gamma-decay - not used
                 rp_reward_threshold=0.1,  # r.prediction: abs.rewards values bigger than this are considered non-zero
                 rp_sequence_size=3,):  # r.prediction sampling
        """

        Args:
            env:                    envirionment instance.
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
            replay_memory_size:     in number of experiences
            replay_rollout_length:  off-policy rollout length
            use_off_policy_aac:     use full AAC off policy training instead of Value-replay
            use_reward_prediction:  use aux. off-policy reward prediction task
            use_pixel_control:      use aux. off-policy pixel control task
            use_value_replay:       use aux. off-policy value replay task (not used, if use_off_policy_aac=True)
            use_rebalanced_replay:  NOT USED
            rebalance_skewness:     NOT USED
            rp_lambda:              reward prediction loss weight
            pc_lambda:              pixel control loss weight
            vr_lambda:              value replay loss weight
            off_aac_lambda:         off-policy AAC loss weight
            gamma_pc:               NOT USED
            rp_reward_threshold:    reward prediction task classification threshold, above which reward is 'non-zero'
            rp_sequence_size:       reward prediction sample size, in number of experiences
        """
        self.log = log
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)
        self.log.debug('AAC_{}_rnd_seed:{}, log_u_sample_(0,1]x5: {}'.
                       format(task, random_seed, log_uniform([1e-10,1], 5)))

        ob_space_type = BTgymMultiSpace
        try:
            assert type(env.observation_space) == ob_space_type

        except:
            raise TypeError('AAC_{}: expected environment observation space of type {}, got: {}'.
                            format(self.task, ob_space_type, type(env.observation_space)))

        self.env = env
        self.task = task
        self.policy_class = policy_config['class_ref']
        self.policy_kwargs = policy_config['kwargs']

        # AAC specific:
        self.model_gamma = model_gamma  # decay
        self.model_gae_lambda = model_gae_lambda  # general advantage estimator lambda
        self.model_beta = log_uniform(model_beta, 1)  # entropy reg.

        # Optimizer
        self.opt_max_train_steps = opt_max_train_steps
        self.opt_learn_rate = log_uniform(opt_learn_rate, 1)

        if opt_end_learn_rate is None:
            self.opt_end_learn_rate = self.opt_learn_rate
        else:
            self.opt_end_learn_rate = opt_end_learn_rate

        if opt_decay_steps is None:
            self.opt_decay_steps = self.opt_max_train_steps
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

        # UNREAL specific:
        self.off_aac_lambda = off_aac_lambda
        self.rp_lambda = rp_lambda
        self.pc_lambda = log_uniform(pc_lambda, 1)
        self.vr_lambda = vr_lambda
        self.gamma_pc = gamma_pc
        self.replay_memory_size = replay_memory_size
        if replay_rollout_length is not None:
            self.replay_rollout_length = replay_rollout_length
        else:
            self.replay_rollout_length = rollout_length
        self.rp_sequence_size = rp_sequence_size
        self.rp_reward_threshold = rp_reward_threshold

        # On/off switchers for off-policy training and auxiliary tasks:
        self.use_off_policy_aac = use_off_policy_aac
        self.use_reward_prediction = use_reward_prediction
        self.use_pixel_control = use_pixel_control
        if use_off_policy_aac:
            self.use_value_replay = False  # v-replay is redundant in this case
        else:
            self.use_value_replay = use_value_replay
        self.use_rebalanced_replay = use_rebalanced_replay
        self.rebalance_skewness = rebalance_skewness

        self.use_any_aux_tasks = use_value_replay or use_pixel_control or use_reward_prediction
        self.use_memory = self.use_any_aux_tasks or self.use_off_policy_aac

        self.log.info(
            'AAC_{}: learn_rate: {:1.6f}, entropy_beta: {:1.6f}, pc_lambda: {:1.8f}.'.
                format(self.task, self.opt_learn_rate, self.model_beta, self.pc_lambda))

        #self.log.info(
        #    'AAC_{}: max_steps: {}, decay_steps: {}, end_rate: {:1.6f},'.
        #        format(self.task, self.opt_max_train_steps, self.opt_decay_steps, self.opt_end_learn_rate))

        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        # Update policy configuration
        self.policy_kwargs.update(
            {
                'ob_space': env.observation_space.shape,
                'ac_space': env.action_space.n,
                'rp_sequence_size': self.rp_sequence_size,
            }
        )
        # Start building graph:

        # PS:
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            self.network = self.make_policy('global')

        # Worker:
        with tf.device(worker_device):
            self.local_network = pi = self.make_policy('local')

            # Meant for Batch-norm layers:
            pi.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='.*local.*')
            self.log.debug('AAC_{}: local_network_upd_ops_collection:\n{}'.format(self.task, pi.update_ops))

            self.log.debug('\nAAC_{}: local_network_var_list_to_save:'.format(self.task))
            for v in pi.var_list:
                self.log.debug('{}: {}'.format(v.name, v.get_shape()))

            #  Learning rate annealing:
            learn_rate = tf.train.polynomial_decay(
                self.opt_learn_rate,
                self.global_step + 1,
                self.opt_decay_steps,
                self.opt_end_learn_rate,
                power=1,
                cycle=False,
            )

            # On-policy AAC loss definition:
            self.on_pi_act_target = tf.placeholder(tf.float32, [None, env.action_space.n], name="on_policy_action_pl")
            self.on_pi_adv_target = tf.placeholder(tf.float32, [None], name="on_policy_advantage_pl")
            self.on_pi_r_target = tf.placeholder(tf.float32, [None], name="on_policy_return_pl")

            on_pi_loss, on_pi_summaries = aac_loss_def(
                act_target=self.on_pi_act_target,
                adv_target=self.on_pi_adv_target,
                r_target=self.on_pi_r_target,
                pi_logits=pi.on_logits,
                pi_vf=pi.on_vf,
                entropy_beta=self.model_beta,
                name='on_policy/aac',
                verbose=True
            )

            # Start accumulating total loss:
            self.loss = on_pi_loss
            model_summaries = on_pi_summaries

            # Off-policy batch size:
            off_bs = tf.to_float(tf.shape(pi.off_state_in[list(pi.on_state_in.keys())[0]])[0])

            if self.use_rebalanced_replay:
                # Simplified importance-sampling bias correction:
                rebalanced_replay_weight = self.rebalance_skewness / off_bs

            else:
                rebalanced_replay_weight = 1.0

            # Off policy training:
            self.off_pi_act_target = tf.placeholder(
                tf.float32, [None, env.action_space.n], name="off_policy_action_pl")
            self.off_pi_adv_target = tf.placeholder(
                tf.float32, [None], name="off_policy_advantage_pl")
            self.off_pi_r_target = tf.placeholder(
                tf.float32, [None], name="off_policy_return_pl")

            if self.use_off_policy_aac:
                # Off-policy PPO loss graph mirrors on-policy:
                off_ppo_loss, off_ppo_summaries = aac_loss_def(
                    act_target=self.off_pi_act_target,
                    adv_target=self.off_pi_adv_target,
                    r_target=self.off_pi_r_target,
                    pi_logits=pi.off_logits,
                    pi_vf=pi.off_vf,
                    entropy_beta=self.model_beta,
                    name='off_policy/aac',
                    verbose=False
                )
                self.loss = self.loss + self.off_aac_lambda * rebalanced_replay_weight * off_ppo_loss
                model_summaries += off_ppo_summaries

            if self.use_pixel_control:
                # Pixel control loss:
                self.pc_action = tf.placeholder(tf.float32, [None, env.action_space.n], name="pc_action")
                self.pc_target = tf.placeholder(tf.float32, [None, None, None], name="pc_target")

                pc_loss, pc_summaries = pc_loss_def(
                    actions=self.pc_action,
                    targets=self.pc_target,
                    pi_pc_q=pi.pc_q,
                    name='off_policy/pixel_control',
                    verbose=True
                )
                self.loss = self.loss + self.pc_lambda * rebalanced_replay_weight * pc_loss
                # Add specific summary:
                model_summaries += pc_summaries

            if self.use_value_replay:
                # Value function replay loss:
                self.vr_target = tf.placeholder(tf.float32, [None], name="vr_target")
                vr_loss, vr_summaries = value_fn_loss_def(
                    r_target=self.vr_target,
                    pi_vf=pi.vr_value,
                    name='off_policy/value_replay',
                    verbose=True
                )
                self.loss = self.loss + self.vr_lambda * rebalanced_replay_weight * vr_loss
                model_summaries += vr_summaries

            if self.use_reward_prediction:
                # Reward prediction loss:
                self.rp_target = tf.placeholder(tf.float32, [1,3], name="rp_target")

                rp_loss, rp_summaries = rp_loss_def(
                    rp_targets=self.rp_target,
                    pi_rp_logits=pi.rp_logits,
                    name='off_policy/reward_prediction',
                    verbose=True
                )
                self.loss = self.loss + self.rp_lambda * rp_loss
                model_summaries += rp_summaries

            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # Copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))

            # Since every observation mod. has same batch size - just take  first key in a row:
            self.inc_step = self.global_step.assign_add(tf.shape(pi.on_state_in[list(pi.on_state_in.keys())[0]])[0])

            # Each worker gets a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(learn_rate, epsilon=1e-5)

            #opt = tf.train.RMSPropOptimizer(
            #    learning_rate=learn_rate,
            #    decay=0.99,
            #    momentum=0.0,
            #    epsilon=1e-8,
            #)

            #self.train_op = tf.group(*pi.update_ops, opt.apply_gradients(grads_and_vars), self.inc_step)
            #self.train_op = tf.group(opt.apply_gradients(grads_and_vars), self.inc_step)
            self.train_op = opt.apply_gradients(grads_and_vars)

            # Add model-wide statistics:
            with tf.name_scope('model'):
                model_summaries += [
                    tf.summary.scalar("grad_global_norm", tf.global_norm(grads)),
                    tf.summary.scalar("var_global_norm", tf.global_norm(pi.var_list)),
                    tf.summary.scalar("learn_rate", learn_rate),
                    tf.summary.scalar("total_loss", self.loss),
                ]

            self.summary_writer = None
            self.local_steps = 0

            self.log.debug('AAC_{}: train op defined'.format(self.task))

            # Model stat. summary:
            self.model_summary_op = tf.summary.merge(model_summaries, name='model_summary')

            # Episode-related summaries:
            self.ep_summary = dict(
                # Summary placeholders
                render_human_pl=tf.placeholder(tf.uint8, [None, None, None, 3]),
                render_model_input_pl=tf.placeholder(tf.uint8, [None, None, None, 3]),
                render_episode_pl=tf.placeholder(tf.uint8, [None, None, None, 3]),
                render_atari_pl=tf.placeholder(tf.uint8, [None, None, None, 1]),
                total_r_pl=tf.placeholder(tf.float32, ),
                cpu_time_pl=tf.placeholder(tf.float32, ),
                final_value_pl=tf.placeholder(tf.float32, ),
                steps_pl=tf.placeholder(tf.int32, ),
            )
            # Environmnet rendering:
            self.ep_summary['render_op'] = tf.summary.merge(
                [
                    tf.summary.image('human', self.ep_summary['render_human_pl']),
                    tf.summary.image('model_input', self.ep_summary['render_model_input_pl']),
                    tf.summary.image('episode', self.ep_summary['render_episode_pl']),
                ],
                name='render'
            )
            # For Atari:
            self.ep_summary['test_render_op'] = tf.summary.image("model/state", self.ep_summary['render_atari_pl'])

            # Episode stat. summary:
            self.ep_summary['stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode/total_reward', self.ep_summary['total_r_pl']),
                    tf.summary.scalar('episode/cpu_time_sec', self.ep_summary['cpu_time_pl']),
                    tf.summary.scalar('episode/final_value', self.ep_summary['final_value_pl']),
                    tf.summary.scalar('episode/env_steps', self.ep_summary['steps_pl'])
                ],
                name='episode'
            )
            self.ep_summary['test_stat_op'] = tf.summary.merge(
                [
                    tf.summary.scalar('episode/total_reward', self.ep_summary['total_r_pl']),
                    tf.summary.scalar('episode/steps', self.ep_summary['steps_pl'])
                ],
                name='episode_atari'
            )

            # Make runner:
            # `rollout_length` represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            self.runner = RunnerThread(
                env,
                pi,
                task,
                self.rollout_length,  # ~20
                self.episode_summary_freq,
                self.env_render_freq,
                self.test_mode,
                self.ep_summary
            )
            # Make rollouts provider method:
            self.get_rollout = make_rollout_getter(self.runner.queue)

            # Make replay memory:
            if self.use_memory:
                self.memory = Memory(
                    history_size=self.replay_memory_size,
                    max_sample_size=self.replay_rollout_length,
                    reward_threshold=self.rp_reward_threshold,
                    task=self.task,
                    log=self.log,
                    rollout_getter=self.get_rollout
                )

            self.log.debug('AAC_{}: init() done'.format(self.task))

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

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)  # starting runner thread
        self.summary_writer = summary_writer

    def process_rp(self, rp_experience_frames):
        """
        Estimates reward prediction target.
        Tuned for Atari visual input
        Returns feed dictionary for `reward prediction` loss estimation subgraph.
        """
        rollout = Rollout()

        # Remove last frame:
        last_frame = rp_experience_frames.pop()

        # Make remaining a rollout to get 'states' batch:
        rollout.add_memory_sample(rp_experience_frames)
        batch = rollout.process(gamma=1)

        # One hot vector for target reward (i.e. reward taken from last of sampled frames):
        r = last_frame['reward']
        rp_t = [0.0, 0.0, 0.0]
        if r > self.rp_reward_threshold:
            rp_t[1] = 1.0  # positive [010]

        elif r < - self.rp_reward_threshold:
            rp_t[2] = 1.0  # negative [001]

        else:
            rp_t[0] = 1.0  # zero [100]

        feeder = feed_dict_from_nested(self.local_network.rp_state_in, batch['state'])
        feeder.update({self.rp_target: np.asarray([rp_t])})
        return feeder

    def process_vr(self, batch):
        """
        Returns feed dictionary for `value replay` loss estimation subgraph.
        """
        if not self.use_off_policy_aac:  # use single pass of network on same off-policy batch
            feeder = feed_dict_from_nested(self.local_network.vr_state_in, batch['state'])
            feeder.update(feed_dict_rnn_context(self.local_network.vr_lstm_state_pl_flatten, batch['context']))
            feeder.update({self.local_network.vr_a_r_in: batch['last_action_reward'], self.vr_target: batch['r']})
        else:
            feeder = {self.vr_target: batch['r']}  # redundant actually :)
        return feeder

    def process_pc(self, batch):
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

    def _fill_replay_memory(self, sess):
        """
        Fills replay memory with initial experiences.
        Supposed to be called by parent worker() just before training begins.
        """
        if self.use_memory:
            sess.run(self.sync)
            while not self.memory.is_full():
                rollout = self.get_rollout()
                self.memory.add_rollout(rollout)
            self.log.info('AAC_{}: replay memory filled.'.format(self.task))

    def process(self, sess):
        """
        Grabs a on_policy_rollout that's been produced by the thread runner,
        samples off_policy rollout[s] from replay memory and updates the parameters.
        The update is then sent to the parameter server.
        """

        # Copy weights from shared to local new_policy:
        sess.run(self.sync)

        # Get and process rollout for on-policy train step:
        on_policy_rollout = self.get_rollout()
        on_policy_batch = on_policy_rollout.process(gamma=self.model_gamma, gae_lambda=self.model_gae_lambda)

        # Feeder for on-policy AAC loss estimation graph:
        feed_dict = feed_dict_from_nested(self.local_network.on_state_in, on_policy_batch['state'])
        feed_dict.update(feed_dict_rnn_context(self.local_network.on_lstm_state_pl_flatten, on_policy_batch['context']))
        feed_dict.update(
            {
                self.local_network.on_a_r_in: on_policy_batch['last_action_reward'],
                self.on_pi_act_target: on_policy_batch['action'],
                self.on_pi_adv_target: on_policy_batch['advantage'],
                self.on_pi_r_target: on_policy_batch['r'],
                self.local_network.train_phase: True,
            }
        )

        if self.use_memory:
            # Get sample from replay memory:
            if self.use_rebalanced_replay:
                off_policy_sample = self.memory.sample_priority(
                    self.replay_rollout_length,
                    skewness=self.rebalance_skewness,
                    exact_size=False
                )
            else:
                off_policy_sample = self.memory.sample_uniform(self.replay_rollout_length)

            off_policy_rollout = Rollout()
            off_policy_rollout.add_memory_sample(off_policy_sample)
            off_policy_batch = off_policy_rollout.process(gamma=self.model_gamma, gae_lambda=self.model_gae_lambda)

            # Feeder for off-policy AAC loss estimation graph:
            off_policy_feed_dict = feed_dict_from_nested(self.local_network.off_state_in, off_policy_batch['state'])
            off_policy_feed_dict.update(
                feed_dict_rnn_context(self.local_network.off_lstm_state_pl_flatten, off_policy_batch['context']))
            off_policy_feed_dict.update(
                {
                    self.local_network.off_a_r_in: off_policy_batch['last_action_reward'],
                    self.off_pi_act_target: off_policy_batch['action'],
                    self.off_pi_adv_target: off_policy_batch['advantage'],
                    self.off_pi_r_target: off_policy_batch['r'],
                }
            )
            feed_dict.update(off_policy_feed_dict)

            # Update with reward prediction subgraph:
            if self.use_reward_prediction:
                # Rebalanced 50/50 sample for RP:
                rp_sample = self.memory.sample_priority(self.rp_sequence_size, skewness=2, exact_size=True)
                feed_dict.update(self.process_rp(rp_sample))

            # Pixel control ...
            if self.use_pixel_control:
                feed_dict.update(self.process_pc(off_policy_batch))

            # VR...
            if self.use_value_replay:
                feed_dict.update(self.process_vr(off_policy_batch))

            # Save on_policy_rollout to replay memory:
            self.memory.add_rollout(on_policy_rollout)

        # Every worker writes model summaries:
        should_compute_summary =\
            self.local_steps % self.model_summary_freq == 0

        fetches = [self.train_op]

        if should_compute_summary:
            fetches = [self.train_op, self.model_summary_op, self.inc_step]
        else:
            fetches = [self.train_op, self.inc_step]

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[-2]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1

        #for k, v in feed_dict.items():
        #    try:
        #        print(k, v.shape)
        #    except:
        #        print(k, type(v))